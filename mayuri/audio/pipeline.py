"""
mayuri/audio/pipeline.py
─────────────────────────
Orchestrates the post-wake audio pipeline:

  wake detected → record command → VAD → STT → agent → TTS

This is the "hot path" that only runs after "Mayuri" is said.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sounddevice as sd
from loguru import logger

from mayuri.config import settings
from mayuri.audio.vad import SileroVAD
from mayuri.stt.transcriber import WhisperTranscriber
from mayuri.tts.synthesizer import PiperTTS
from mayuri.api.ws.events import EventBus


class AudioPipeline:
    """Runs the full audio pipeline triggered by WakeWordService."""

    def __init__(self) -> None:
        self._vad = SileroVAD()
        self._stt = WhisperTranscriber()
        self._tts = PiperTTS()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio")

    async def run(self, wake_service: object, stop: asyncio.Event) -> None:
        """Main loop — waits for wake events, then processes commands."""
        logger.info("Audio pipeline ready.")

        while not stop.is_set():
            # Block until wake word fires
            await wake_service.wait_for_wake()
            logger.info("🔊 Listening for command …")

            # Play a short confirmation chime so user knows Mayuri heard them
            await EventBus.emit("state", {"status": "listening"})

            # Record and transcribe the command
            loop = asyncio.get_running_loop()
            pcm_bytes = await loop.run_in_executor(
                self._executor, self._record_until_silence
            )

            if not pcm_bytes:
                logger.warning("No speech captured after wake word.")
                continue

            text = await loop.run_in_executor(
                self._executor, self._stt.transcribe_bytes, pcm_bytes
            )

            if not text.strip():
                logger.warning("STT returned empty transcript.")
                continue

            logger.info(f"📝 Transcribed: '{text}'")
            await EventBus.emit("transcript", {"text": text})

            # Send to agent (imports lazily to avoid circular)
            from mayuri.agent.graph import MayuriAgent
            agent = MayuriAgent()
            response_text = await agent.run(user_input=text)

            logger.info(f"🤖 Agent response: '{response_text[:80]}…'")
            await EventBus.emit("response", {"text": response_text})

            # Speak the response
            await loop.run_in_executor(
                self._executor, self._tts.speak, response_text
            )

    # ── Private (blocking) ────────────────────────────────────────────────────

    def _record_until_silence(self) -> bytes | None:
        """
        Record microphone until VAD detects end of speech.
        Runs in a thread — blocks.
        Returns raw 16-bit PCM bytes.
        """
        self._vad.reset()
        chunk_size = SileroVAD.CHUNK_SAMPLES
        rate = SileroVAD.SAMPLE_RATE
        max_seconds = 30  # safety limit

        collected: list[np.ndarray] = []

        with sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype="int16",
            blocksize=chunk_size,
            device=settings.audio_device_index,
        ) as stream:
            samples_recorded = 0
            while samples_recorded < rate * max_seconds:
                data, _ = stream.read(chunk_size)
                chunk = np.frombuffer(data, dtype=np.int16)
                samples_recorded += len(chunk)

                utterance = self._vad.process(chunk)
                if utterance is not None:
                    return utterance

        logger.warning("Max recording duration reached without VAD end.")
        return None
