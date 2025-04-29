#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import time
import threading
from typing import AsyncGenerator, Optional, Dict, Any
from urllib.parse import urlencode

from loguru import logger
import websocket

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


class AssemblyAISTTService(STTService):
    def __init__(
        self,
        *,
        api_key: str,
        sample_rate: Optional[int] = None,
        encoding: str = "pcm_s16le",
        language=Language.EN,  # Only English is supported for Realtime
        use_direct_websocket: bool = True,  # Use direct websocket instead of SDK
        api_endpoint_base_url: str = "wss://streaming.assemblyai.com/v3/ws",
        formatted_finals: bool = True,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._use_direct_websocket = use_direct_websocket
        self._api_endpoint_base_url = api_endpoint_base_url
        
        # WebSocket related variables
        self._ws_app = None
        self._ws_thread = None
        self._stop_event = threading.Event()
        self._audio_buffer = bytearray()
        self._buffer_lock = threading.Lock()
        
        self._settings = {
            "encoding": encoding,
            "language": language,
            "sample_rate": sample_rate or 16000,  # Default to 16kHz if not specified
            "formatted_finals": formatted_finals,
        }
        
        # Custom metrics tracking
        self._ttfb_start_time = 0
        self._processing_start_time = 0
        self._ttfb_reported = False
        self._active_speech = False
        self._lock = asyncio.Lock()
        
        # Queue for handling transcription results from websocket thread
        self._result_queue = asyncio.Queue()
        self._connected = False

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

    async def start_ttfb_custom_metrics(self):
        """Start measuring time to first byte metrics when speech is detected."""
        async with self._lock:
            self._ttfb_start_time = time.time()
            self._ttfb_reported = False
            logger.debug(f"{self.name}: Started TTFB metrics tracking at {self._ttfb_start_time}")

    async def start_processing_custom_metrics(self):
        """Start measuring processing time metrics when speech stops."""
        async with self._lock:
            self._processing_start_time = time.time()
            logger.debug(f"{self.name}: Started processing metrics tracking at {self._processing_start_time}")

    async def emit_ttfb_custom_metrics(self):
        """Emit TTFB metrics when first partial transcript is received."""
        async with self._lock:
            if self._ttfb_start_time == 0 or self._ttfb_reported:
                return
            
            elapsed = time.time() - self._ttfb_start_time
            logger.debug(f"{self.name}: TTFB metrics - {elapsed:.3f}s")
            
            # Create and emit metrics frame
            ttfb = TTFBMetricsData(processor=self.name, value=elapsed)
            await self.push_frame(MetricsFrame(data=[ttfb]))
            
            self._ttfb_reported = True

    async def emit_processing_custom_metrics(self):
        """Emit processing metrics when final transcript is received."""
        async with self._lock:
            if self._processing_start_time == 0:
                return
            
            elapsed = time.time() - self._processing_start_time
            logger.debug(f"{self.name}: Processing metrics - {elapsed:.3f}s")
            
            # Create and emit metrics frame
            processing = ProcessingMetricsData(processor=self.name, value=elapsed)
            await self.push_frame(MetricsFrame(data=[processing]))
            
            # Reset processing metrics state after final processing
            self._processing_start_time = 0

    async def _process_result_queue(self):
        """Process transcription results from the websocket thread."""
        while True:
            try:
                result = await self._result_queue.get()
                if result is None:  # None is our signal to stop
                    break
                    
                msg_type = result.get('type')
                timestamp = time_now_iso8601()
                
                if msg_type == 'Partial':
                    text = result.get('text', '')
                    if text:
                        # For interim transcripts, create an InterimTranscriptionFrame
                        frame = InterimTranscriptionFrame(
                            text, "", timestamp, self._settings["language"]
                        )
                        await self.push_frame(frame)
                        
                        # If we have active speech and haven't reported TTFB yet, do so now
                        if self._active_speech and not self._ttfb_reported:
                            await self.emit_ttfb_custom_metrics()
                            
                elif msg_type == 'Final':
                    text = result.get('text', '')
                    if text:
                        # For final transcripts, create a TranscriptionFrame and emit processing metrics
                        frame = TranscriptionFrame(
                            text, "", timestamp, self._settings["language"]
                        )
                        await self.push_frame(frame)
                        
                        # Emit processing metrics for final transcript
                        await self.emit_processing_custom_metrics()
                        self._active_speech = False
                            
                elif msg_type == 'Error':
                    error_message = result.get('error', 'Unknown error from AssemblyAI')
                    logger.error(f"{self}: An error occurred: {error_message}")
                    await self.push_frame(ErrorFrame(error_message))
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{self}: Error processing result: {e}")
                await self.push_frame(ErrorFrame(str(e)))
            finally:
                self._result_queue.task_done()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process an audio chunk for STT transcription.

        This method streams the audio data to AssemblyAI for real-time transcription.
        Transcription results are handled asynchronously via callback functions.

        :param audio: Audio data as bytes
        :yield: None (transcription frames are pushed via self.push_frame in callbacks)
        """
        if self._connected:
            # Add audio to buffer - will be sent by websocket thread
            with self._buffer_lock:
                self._audio_buffer.extend(audio)
        yield None

    async def _connect(self):
        """Establish a connection to the AssemblyAI real-time transcription service.

        This method sets up the necessary callback functions and initializes the
        websocket connection to AssemblyAI.
        """
        if self._connected:
            return
            
        # Start the result queue processor
        self._result_processor_task = asyncio.create_task(self._process_result_queue())
        
        # Reset the stop event
        self._stop_event.clear()
        
        # Build the connection URL with parameters
        connection_params = {
            "sample_rate": self._settings["sample_rate"],
            "encoding": self._settings["encoding"],
        }
        api_endpoint = f"{self._api_endpoint_base_url}?{urlencode(connection_params)}"
        
        def on_open(ws):
            """Callback for when the connection to AssemblyAI is opened."""
            logger.info(f"{self}: Connected to AssemblyAI WebSocket")
            self._connected = True
            
            # Start sending audio data in a separate thread
            def stream_audio():
                logger.debug("Starting audio streaming thread...")
                while not self._stop_event.is_set():
                    try:
                        # Get audio from buffer
                        with self._buffer_lock:
                            if len(self._audio_buffer) > 0:
                                audio_data = bytes(self._audio_buffer)
                                self._audio_buffer.clear()
                                # Send audio data as binary message
                                ws.send(audio_data, websocket.ABNF.OPCODE_BINARY)
                        # Sleep a tiny bit to avoid spinning
                        time.sleep(0.01)
                    except Exception as e:
                        logger.error(f'Error streaming audio: {e}')
                        # If send fails, likely means connection is closed
                        break
                logger.debug("Audio streaming thread stopped.")

            self._audio_thread = threading.Thread(target=stream_audio)
            self._audio_thread.daemon = True
            self._audio_thread.start()

        def on_message(ws, message):
            """Callback for handling incoming transcription data."""
            try:
                data = json.loads(message)
                # Put the result in the queue for processing in the main event loop
                asyncio.run_coroutine_threadsafe(
                    self._result_queue.put(data), 
                    self.get_event_loop()
                )
            except json.JSONDecodeError:
                logger.error(f"Received non-JSON message: {message}")
            except Exception as e:
                logger.error(f'Error handling message: {e}')

        def on_error(ws, error):
            """Callback for handling errors from AssemblyAI."""
            logger.error(f"{self}: WebSocket Error: {error}")
            
            # Put error in queue
            error_data = {"type": "Error", "error": str(error)}
            asyncio.run_coroutine_threadsafe(
                self._result_queue.put(error_data),
                self.get_event_loop()
            )
            
            # Signal stop
            self._stop_event.set()

        def on_close(ws, close_status_code, close_msg):
            """Callback for when the connection to AssemblyAI is closed."""
            logger.info(f"{self}: WebSocket Disconnected: Status={close_status_code}, Msg={close_msg}")
            self._connected = False
            
            # Signal audio thread to stop
            self._stop_event.set()
            
            # Put close event in queue
            close_data = {"type": "Close"}
            asyncio.run_coroutine_threadsafe(
                self._result_queue.put(close_data),
                self.get_event_loop()
            )

        # Create WebSocketApp
        self._ws_app = websocket.WebSocketApp(
            api_endpoint,
            header={'Authorization': self._api_key},
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Run WebSocketApp in a separate thread
        self._ws_thread = threading.Thread(target=self._ws_app.run_forever)
        self._ws_thread.daemon = True
        self._ws_thread.start()
        
        # Wait a bit for the connection to establish
        for _ in range(10):  # Wait up to 1 second
            if self._connected:
                break
            await asyncio.sleep(0.1)

    async def _disconnect(self):
        """Disconnect from the AssemblyAI service and clean up resources."""
        if not self._connected:
            return
            
        logger.info(f"{self}: Disconnecting from AssemblyAI WebSocket")
        
        # Signal audio thread to stop
        self._stop_event.set()
        
        # Send termination message to the server
        if self._ws_app and self._ws_app.sock and self._ws_app.sock.connected:
            try:
                terminate_message = {"type": "Terminate"}
                logger.debug(f"Sending termination message: {json.dumps(terminate_message)}")
                self._ws_app.send(json.dumps(terminate_message))
                # Give a moment for messages to process before forceful close
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error sending termination message: {e}")
                
        # Close the WebSocket connection
        if self._ws_app:
            self._ws_app.close()
            
        # Wait for WebSocket thread to finish
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2.0)
            
        # Signal the result processor to stop
        await self._result_queue.put(None)
        if hasattr(self, '_result_processor_task'):
            await self._result_processor_task
            
        # Clean up resources
        self._connected = False
        self._ws_app = None
        self._ws_thread = None
        self._active_speech = False
        
        # Clear audio buffer
        with self._buffer_lock:
            self._audio_buffer.clear()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames from the pipeline.
        
        This method enables the service to react to user started/stopped speaking frames
        to start and manage metrics appropriately.
        """
        await super().process_frame(frame, direction)
        
        if isinstance(frame, UserStartedSpeakingFrame):
            # Start TTFB metrics when speech is detected
            logger.debug(f"{self.name}: Speech detected, starting TTFB metrics")
            self._active_speech = True
            await self.start_ttfb_custom_metrics()
            
            # We don't need the standard metrics calls anymore
            #await self.start_ttfb_metrics()
            #await self.start_processing_metrics()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Start processing metrics when speech stops
            logger.debug(f"{self.name}: Speech stopped, starting processing metrics")
            
            # We don't need the standard metrics call anymore
            #await self.stop_all_metrics()
            
            if self._active_speech:
                # Start processing metrics from when speech stopped
                await self.start_processing_custom_metrics()
