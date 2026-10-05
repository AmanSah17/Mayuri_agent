"""
mayuri/audio/capture.py
────────────────────────
Microphone capture helpers.

Provides a simple generator that yields fixed-size PCM chunks from
the default (or configured) audio device using sounddevice.

Why a generator?
  Generators compose cleanly: wake_service, VAD, and pipeline all
  just iterate over chunks without sharing mutable buffer state.
"""

from __future__ import annotations

from collections.abc import Generator
from queue import Queue, Empty

import numpy as np
import sounddevice as sd
from loguru import logger

from mayuri.config import settings


def list_devices() -> str:
    """Return a human-readable list of audio devices."""
    return str(sd.query_devices())


def mic_chunk_generator(
    chunk_samples: int = 512,
    stop_after_silence_s: float | None = None,
) -> Generator[np.ndarray, None, None]:
    """
    Generator: yields numpy arrays of shape (chunk_samples,) dtype int16.

    Parameters
    ----------
    chunk_samples:
        Number of samples per chunk (default 512 ≈ 32 ms @ 16 kHz).
    stop_after_silence_s:
        If set, stops the generator after this many seconds of near-silence.
        Useful for bounded command recording.
    """
    q: Queue[np.ndarray | None] = Queue()

    def callback(indata: np.ndarray, frames: int, time: object, status: object) -> None:
        if status:
            logger.warning(f"sounddevice status: {status}")
        q.put(indata.copy().flatten())

    with sd.InputStream(
        samplerate=settings.audio_sample_rate,
        channels=settings.audio_channels,
        dtype="int16",
        blocksize=chunk_samples,
        device=settings.audio_device_index,
        callback=callback,
    ):
        logger.debug(f"Microphone opened (device={settings.audio_device_index}, "
                     f"rate={settings.audio_sample_rate})")
        while True:
            try:
                chunk = q.get(timeout=1.0)
                if chunk is None:
                    break
                yield chunk
            except Empty:
                continue
