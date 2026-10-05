"""
mayuri/stt/transcriber.py
──────────────────────────
faster-whisper STT wrapper.

faster-whisper uses CTranslate2 and can run on CPU with int8 quantisation
or on CUDA with float16. On a mid-range GPU (e.g., RTX 3060) it transcribes
real-time speech in ~200 ms.

Design decisions:
  - Model is loaded once at startup (lazy, on first call).
  - transcribe_bytes() accepts raw PCM (16-bit, 16 kHz, mono).
  - Strips "Mayuri" from the beginning of the transcript — the user said
    the wake word but we don't want it in the LLM prompt.
"""

from __future__ import annotations

import io
import re
import wave
from functools import cached_property

import numpy as np
from loguru import logger

from mayuri.config import settings


class WhisperTranscriber:
    """faster-whisper based transcriber."""

    SAMPLE_RATE = 16_000

    # ── Lazy model load ───────────────────────────────────────────────────────

    @cached_property
    def _model(self) -> object:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("faster-whisper not installed: pip install faster-whisper") from exc

        device = settings.whisper_device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        compute = settings.whisper_compute_type
        if device == "cpu" and compute == "float16":
            compute = "int8"  # float16 unsupported on CPU

        logger.info(
            f"Loading Whisper '{settings.whisper_model_size}' on {device}/{compute} …"
        )
        from faster_whisper import WhisperModel
        model = WhisperModel(
            settings.whisper_model_size,
            device=device,
            compute_type=compute,
        )
        logger.info("Whisper ready.")
        return model

    # ── Public API ────────────────────────────────────────────────────────────

    def transcribe_bytes(self, pcm_bytes: bytes) -> str:
        """
        Transcribe raw PCM bytes → text.

        Parameters
        ----------
        pcm_bytes:
            Raw signed 16-bit, 16 kHz, mono PCM audio.
        """
        # faster-whisper accepts a file-like object (WAV) or numpy float32 array
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,   # built-in VAD as extra filter
        )

        text = " ".join(seg.text.strip() for seg in segments)
        text = self._strip_wake_word(text)
        return text.strip()

    def transcribe_file(self, path: str) -> str:
        """Convenience: transcribe from a WAV file path."""
        segments, _ = self._model.transcribe(path, language="en", beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_wake_word(text: str) -> str:
        """Remove leading 'Mayuri' / 'Hey Mayuri' from transcript."""
        pattern = re.compile(r"^\s*(hey\s+)?mayuri[,.]?\s*", re.IGNORECASE)
        return pattern.sub("", text)
