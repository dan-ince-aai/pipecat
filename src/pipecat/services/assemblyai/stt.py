#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import time
from typing import AsyncGenerator, Optional, Dict

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import assemblyai as aai
    from assemblyai import AudioEncoding
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use AssemblyAI, you need to `pip install pipecat-ai[assemblyai]`.")
    raise Exception(f"Missing module: {e}")


class AssemblyAISTTService(STTService):
    def __init__(
        self,
        *,
        api_key: str,
        sample_rate: Optional[int] = None,
        encoding: AudioEncoding = AudioEncoding("pcm_s16le"),
        language=Language.EN,  # Only English is supported for Realtime
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        aai.settings.api_key = api_key
        self._transcriber: Optional[aai.RealtimeTranscriber] = None

        self._settings = {
            "encoding": encoding,
            "language": language,
        }
        
        # Custom metrics tracking
        self._metrics_started = False
        self._ttfb_start_time = 0
        self._processing_start_time = 0
        self._ttfb_reported = False
        self._active_speech = False
        self._lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        """Indicate that this service supports metrics generation."""
        return True

    async def set_language(self, language: Language):
        logger.info(f"Switching STT language to: [{language}]")
        self._settings["language"] = language

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def custom_start_metrics(self):
        """Custom implementation to start measuring metrics for real API interaction."""
        async with self._lock:
            self._ttfb_start_time = time.time()
            self._processing_start_time = time.time()
            self._ttfb_reported = False
            self._metrics_started = True
            logger.debug(f"{self.name}: Started custom metrics tracking at {self._ttfb_start_time}")

    async def custom_stop_all_metrics(self):
        """Stop all custom metrics tracking."""
        async with self._lock:
            self._metrics_started = False
            self._ttfb_start_time = 0
            self._processing_start_time = 0
            self._ttfb_reported = False
            logger.debug(f"{self.name}: Stopped all custom metrics tracking")

    async def emit_ttfb_metrics(self):
        """Emit TTFB metrics using the custom tracking."""
        async with self._lock:
            if not self._metrics_started or self._ttfb_start_time == 0 or self._ttfb_reported:
                return
            
            elapsed = time.time() - self._ttfb_start_time
            logger.debug(f"{self.name}: TTFB metrics - {elapsed:.3f}s")
            
            # Create and emit metrics frame
            ttfb = TTFBMetricsData(processor=self.name, value=elapsed)
            await self.push_frame(MetricsFrame(data=[ttfb]))
            
            self._ttfb_reported = True

    async def emit_processing_metrics(self):
        """Emit processing metrics using the custom tracking."""
        async with self._lock:
            if not self._metrics_started or self._processing_start_time == 0:
                return
            
            elapsed = time.time() - self._processing_start_time
            logger.debug(f"{self.name}: Processing metrics - {elapsed:.3f}s")
            
            # Create and emit metrics frame
            processing = ProcessingMetricsData(processor=self.name, value=elapsed)
            await self.push_frame(MetricsFrame(data=[processing]))
            
            # Reset metrics state after final processing
            self._metrics_started = False
            self._processing_start_time = 0
            self._ttfb_start_time = 0

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process an audio chunk for STT transcription.

        This method streams the audio data to AssemblyAI for real-time transcription.
        Transcription results are handled asynchronously via callback functions.

        :param audio: Audio data as bytes
        :yield: None (transcription frames are pushed via self.push_frame in callbacks)
        """
        if self._transcriber:
            # Don't do any metrics here - metrics are handled by custom tracking
            self._transcriber.stream(audio)
        yield None

    async def _connect(self):
        """Establish a connection to the AssemblyAI real-time transcription service.

        This method sets up the necessary callback functions and initializes the
        AssemblyAI transcriber.
        """
        if self._transcriber:
            return

        def on_open(session_opened: aai.RealtimeSessionOpened):
            """Callback for when the connection to AssemblyAI is opened."""
            logger.info(f"{self}: Connected to AssemblyAI")

        def on_data(transcript: aai.RealtimeTranscript):
            """Callback for handling incoming transcription data.

            This function runs in a separate thread from the main asyncio event loop.
            It creates appropriate transcription frames and schedules them to be
            pushed to the next stage of the pipeline in the main event loop.
            """
            if not transcript.text:
                return

            timestamp = time_now_iso8601()

            # Create a coroutine for handling this transcript with metrics
            async def handle_transcript():
                # If we have active metrics and haven't reported TTFB yet, do so now
                if self._active_speech and self._metrics_started and not self._ttfb_reported:
                    await self.emit_ttfb_metrics()
                
                if isinstance(transcript, aai.RealtimeFinalTranscript):
                    # For final transcripts, create a TranscriptionFrame and emit processing metrics
                    frame = TranscriptionFrame(
                        transcript.text, "", timestamp, self._settings["language"]
                    )
                    await self.push_frame(frame)
                    
                    # Emit processing metrics for final transcript
                    if self._active_speech and self._metrics_started:
                        await self.emit_processing_metrics()
                        self._active_speech = False
                else:
                    # For interim transcripts, just create an InterimTranscriptionFrame
                    frame = InterimTranscriptionFrame(
                        transcript.text, "", timestamp, self._settings["language"]
                    )
                    await self.push_frame(frame)

            # Schedule the coroutine to run in the main event loop
            # This is necessary because this callback runs in a different thread
            asyncio.run_coroutine_threadsafe(handle_transcript(), self.get_event_loop())

        def on_error(error: aai.RealtimeError):
            """Callback for handling errors from AssemblyAI.

            Like on_data, this runs in a separate thread and schedules error
            handling in the main event loop.
            """
            logger.error(f"{self}: An error occurred: {error}")

            async def handle_error():
                await self.custom_stop_all_metrics()
                await self.push_frame(ErrorFrame(str(error)))
                
            # Schedule the coroutine to run in the main event loop
            asyncio.run_coroutine_threadsafe(handle_error(), self.get_event_loop())

        def on_close():
            """Callback for when the connection to AssemblyAI is closed."""
            logger.info(f"{self}: Disconnected from AssemblyAI")

            async def handle_close():
                await self.custom_stop_all_metrics()
                
            # Schedule the coroutine to run in the main event loop
            asyncio.run_coroutine_threadsafe(handle_close(), self.get_event_loop())

        self._transcriber = aai.RealtimeTranscriber(
            sample_rate=self.sample_rate,
            encoding=self._settings["encoding"],
            on_data=on_data,
            on_error=on_error,
            on_open=on_open,
            on_close=on_close,
        )
        self._transcriber.connect()

    async def _disconnect(self):
        """Disconnect from the AssemblyAI service and clean up resources."""
        if self._transcriber:
            self._transcriber.close()
            self._transcriber = None
            await self.custom_stop_all_metrics()
            self._active_speech = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames from the pipeline.
        
        This method enables the service to react to user started/stopped speaking frames
        to start and manage metrics appropriately.
        """
        await super().process_frame(frame, direction)
        
        if isinstance(frame, UserStartedSpeakingFrame):
            # Start custom metrics when speech is detected
            logger.debug(f"{self.name}: Speech detected, starting metrics")
            self._active_speech = True
            await self.custom_start_metrics()
            
            # We still call the standard metrics methods for compatibility
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # If user stopped speaking but we haven't emitted metrics,
            # we should stop metrics tracking to prevent hanging metrics
            logger.debug(f"{self.name}: Speech stopped")
            
            # We still call the standard metrics methods for compatibility
            await self.stop_all_metrics()
            
            if self._active_speech and self._metrics_started and not self._ttfb_reported:
                # No transcription was received, so reset metrics
                await self.custom_stop_all_metrics()
                self._active_speech = False
