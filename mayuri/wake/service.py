"""
mayuri/wake/service.py
───────────────────────
Always-on wake-word service.

Continuously reads from the microphone in tiny 80 ms chunks and feeds
them to the WakeWordDetector. When the wake word fires it sets an
asyncio.Event so the AudioPipeline knows to start recording the command.

This service intentionally has ZERO dependency on the LLM, STT, or TTS.
It must be cheap enough to run 24/7.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sounddevice as sd
from loguru import logger

from mayuri.config import settings
from mayuri.wake.detector import WakeWordDetector


class WakeWordService:
    """Manages the always-on microphone + wake-word loop."""

    def __init__(self) -> None:
        self._detector = WakeWordDetector()
        self._detected_event = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wake")

    # ── Public ────────────────────────────────────────────────────────────────

    async def wait_for_wake(self) -> None:
        """Suspend until wake word fires, then clear the event."""
        await self._detected_event.wait()
        self._detected_event.clear()

    async def run(self, stop: asyncio.Event) -> None:
        """
        Main loop — runs until *stop* is set.
        Offloads blocking sounddevice I/O to a thread.
        """
        logger.info(f"🎙️  Wake-word service started — listening for '{settings.wake_word}' …")
        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(self._executor, self._blocking_loop, stop, loop)
        except Exception as exc:
            logger.error(f"Wake-word service error: {exc}")

    # ── Private ───────────────────────────────────────────────────────────────

    def _blocking_loop(self, stop: asyncio.Event, loop: asyncio.AbstractEventLoop) -> None:
        """
        Blocking audio capture loop — runs inside ThreadPoolExecutor.
        Uses sounddevice's RawInputStream for lowest latency.
        """
        chunk = WakeWordDetector.CHUNK_SAMPLES
        rate = WakeWordDetector.SAMPLE_RATE

        with sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype="int16",
            blocksize=chunk,
            device=settings.audio_device_index,
        ) as stream:
            while not stop.is_set():
                data, _ = stream.read(chunk)
                audio = np.frombuffer(data, dtype=np.int16)

                if self._detector.process_chunk(audio):
                    # Signal the async world from this thread
                    loop.call_soon_threadsafe(self._detected_event.set)
                    logger.info("✨ Wake word detected!")
