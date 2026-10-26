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
        clean_text, _, _ = self.transcribe_with_info(pcm_bytes)
        return clean_text

    def transcribe_with_info(self, pcm_bytes: bytes) -> tuple[str, str, bool]:
        """
        Transcribe raw PCM bytes, returning (clean_text, raw_text, has_wake_word).
        """
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=False,  # Audio is already pre-filtered by Silero VAD
            initial_prompt="Mayuri, Hey Mayuri, Jarvis, Hey Jarvis, wake up, please wake up.",
        )

        raw_text = " ".join(seg.text.strip() for seg in segments).strip()
        custom_wake = re.escape(settings.wake_word.replace("_", " ")).strip()
        wake_pattern = rf"\b(mayuri|mayuree|maiyuri|may\s*uri|my\s*uri|mayur|jarvis|hey\s*jarvis|alexa|wake\s*up|please\s*wake\s*up|makeup|speak\s*up|beauty|enjoy\s*this|in\s*charge\s*of\s*this|ajahn|{custom_wake})\b"
        has_wake = bool(re.search(wake_pattern, raw_text, re.IGNORECASE))
        clean_text = self._strip_wake_word(raw_text).strip()
        return clean_text, raw_text, has_wake

    def transcribe_file(self, path: str) -> str:
        """Convenience: transcribe from a WAV file path."""
        segments, _ = self._model.transcribe(
            path,
            language="en",
            beam_size=5,
            vad_filter=False,
            initial_prompt="Mayuri, Hey Mayuri, Jarvis, Hey Jarvis, wake up.",
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_wake_word(text: str) -> str:
        """Remove leading 'Mayuri' / 'Jarvis' / 'Hey Jarvis' / 'wake up' from transcript."""
        custom_wake = re.escape(settings.wake_word.replace("_", " ")).strip()
        pattern = re.compile(
            rf"^\s*(hey\s+|hi\s+)?(mayuri|mayuree|maiyuri|may\s*uri|my\s*uri|mayur|jarvis|alexa|{custom_wake}|ajahn|enjoy\s*this|in\s*charge\s*of\s*this|i'm\s*a\s*beauty)[,.]?\s*(\b(please\s+)?(wake\s*up|makeup|speak\s*up)\b)?\s*[,.]?\s*",
            re.IGNORECASE,
        )
        cleaned = pattern.sub("", text).strip()
        # Also clean standalone 'wake up'
        cleaned = re.sub(r"^\s*(please\s+)?wake\s*up[,.]?\s*", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned
