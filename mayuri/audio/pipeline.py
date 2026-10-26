"""
mayuri/audio/pipeline.py
─────────────────────────
Full-duplex continuous audio pipeline for Mayuri:

  Microphone → Silero VAD → faster-whisper STT → LangGraph Agent → Piper TTS

Features:
  - Unified microphone stream (no competing sounddevice streams)
  - VAD speech segmentation (Silero)
  - Natural wake-word detection ("Mayuri", "Hey Mayuri")
  - Conversational continuity (remembers active conversation for 45 seconds so follow-up speech is heard without repeating "Mayuri")
  - Echo suppression (mutes mic input while TTS is speaking)
  - Real-time WebSocket event emission for UI responsiveness
"""

from __future__ import annotations

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

import numpy as np
import sounddevice as sd
from loguru import logger

from mayuri.api.ws.events import EventBus
from mayuri.audio.vad import SileroVAD
from mayuri.config import settings
from mayuri.stt.transcriber import WhisperTranscriber
from mayuri.tts.synthesizer import PiperTTS


class AudioPipeline:
    """Unified audio engine: microphone capture, VAD, STT, agent execution, and TTS."""

    def __init__(self) -> None:
        self._vad = SileroVAD()
        self._stt = WhisperTranscriber()
        self._tts = PiperTTS()
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="audio")

        self._is_speaking = False
        self._current_task = None
        self._last_active_time: float = 0.0
        self._conversation_timeout: float = 45.0  # seconds

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, wake_service: object | None = None, stop: asyncio.Event | None = None) -> None:
        """Main audio processing loop."""
        if stop is None:
            stop = asyncio.Event()

        logger.info("🎙️  Audio pipeline started — continuous speech listener active.")
        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(self._executor, self._capture_and_process_loop, stop, loop)
        except Exception as exc:
            logger.error(f"Audio pipeline stopped with error: {exc}")

    async def process_text(
        self,
        text: str,
        speak: bool = False,
        session_id: str = "default",
    ) -> str:
        """
        Process user text through the agent (Qwen model) and speak the response if requested.
        """
        logger.info(f"📝 Pipeline processing: '{text}' (session={session_id})")
        await EventBus.emit("transcript", {"text": text})
        await EventBus.emit("state", {"status": "thinking"})

        from mayuri.agent.graph import MayuriAgent

        agent = MayuriAgent()
        response_text = await agent.run(user_input=text, session_id=session_id)

        logger.info(f"🤖 Agent response: '{response_text[:80]}…'")
        await EventBus.emit("response", {"text": response_text})

        if speak and response_text.strip():
            self._is_speaking = True
            await EventBus.emit("state", {"status": "speaking"})
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(self._executor, self._tts.speak, response_text)
            finally:
                self._is_speaking = False
                self._last_active_time = time.time()
                await EventBus.emit("state", {"status": "idle"})
        else:
            await EventBus.emit("state", {"status": "idle"})

        return response_text

    # ── Private ───────────────────────────────────────────────────────────────

    def _capture_and_process_loop(self, stop: asyncio.Event, loop: asyncio.AbstractEventLoop) -> None:
        """Blocking microphone capture and speech recognition worker."""
        chunk_size = SileroVAD.CHUNK_SAMPLES  # 512 samples (~32ms at 16kHz)
        rate = SileroVAD.SAMPLE_RATE
        audio_queue: Queue[np.ndarray | None] = Queue(maxsize=100)

        def audio_callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
            if status:
                logger.debug(f"sounddevice status: {status}")
            if not self._is_speaking:
                try:
                    audio_queue.put_nowait(indata.copy().flatten())
                except Exception:
                    pass  # queue full, drop oldest

        logger.info(f"Opening microphone input stream at {rate}Hz...")
        try:
            with sd.InputStream(
                samplerate=rate,
                channels=1,
                dtype="int16",
                blocksize=chunk_size,
                device=settings.audio_device_index,
                callback=audio_callback,
            ):
                logger.info("🎤 Microphone stream active and listening for speech.")
                while not stop.is_set():
                    try:
                        chunk = audio_queue.get(timeout=0.1)
                    except Empty:
                        continue

                    if self._is_speaking:
                        # Echo suppression: discard mic audio while TTS output is playing
                        self._vad.reset()
                        continue

                    utterance = self._vad.process(chunk)
                    if utterance is not None:
                        # Full utterance detected
                        self._handle_utterance(utterance, loop)

        except Exception as exc:
            logger.error(f"Failed to run microphone stream: {exc}")

    def _handle_utterance(self, pcm_bytes: bytes, loop: asyncio.AbstractEventLoop) -> None:
        """Transcribe utterance and dispatch to agent if addressed to Mayuri or in active dialogue."""
        if len(pcm_bytes) < 16000 * 0.4 * 2:  # Ignore very short blips (<0.4 sec)
            return

        try:
            clean_text, raw_text, has_wake = self._stt.transcribe_with_info(pcm_bytes)
        except Exception as exc:
            logger.warning(f"Whisper STT failed on audio chunk: {exc}")
            return

        if not raw_text.strip():
            return

        logger.info(f"🎙️ Heard: '{raw_text}' (wake_detected={has_wake})")

        # Instant voice interruption / barge-in: "stop", "cancel", "pause", "shut up", "be quiet"
        if re.search(r"\b(stop|cancel|pause|shut\s*up|be\s*quiet|quiet|halt|stop\s*now)\b", raw_text, re.IGNORECASE):
            logger.info("🛑 Voice interruption received! Cutting off speech and resetting.")
            self._tts.stop()
            self._is_speaking = False
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()
            asyncio.run_coroutine_threadsafe(EventBus.emit("state", {"status": "idle"}), loop)
            return

        now = time.time()
        is_in_conversation = (now - self._last_active_time) < self._conversation_timeout

        # Trigger condition:
        # 1. Spoke "Mayuri" / "Jarvis" / "wake up"
        # 2. OR conversation is currently active (follow-up question/command)
        if has_wake or is_in_conversation:
            target_text = clean_text if clean_text.strip() else raw_text
            if not target_text.strip():
                # User just said "Mayuri!" -> greet user
                target_text = "Hello Mayuri"

            logger.info(f"⚡ Command accepted: '{target_text}' (wake={has_wake}, follow_up={is_in_conversation})")
            self._last_active_time = now

            # If previous response is still speaking or generating, immediately interrupt it so the new question takes priority
            if self._is_speaking:
                self._tts.stop()
                self._is_speaking = False
            if self._current_task and not self._current_task.done():
                self._current_task.cancel()

            # Dispatch new command immediately
            self._current_task = asyncio.run_coroutine_threadsafe(
                self.process_text(target_text, speak=True),
                loop,
            )
        else:
            logger.info(f"⚪ Heard audio: '{raw_text}' (Ignored: wake phrase not detected. Say 'Mayuri' or 'Hey Jarvis' to activate)")
