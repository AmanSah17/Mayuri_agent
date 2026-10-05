"""
mayuri/tts/synthesizer.py
──────────────────────────
Piper TTS wrapper.

Piper is a fast, local neural TTS engine (ONNX-based).
It generates WAV audio from text and plays it through the default speaker.

Model download:
  python scripts/install_models.py  (or manual — see README)

Voices: https://github.com/rhasspy/piper/blob/master/VOICES.md
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from functools import cached_property
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import numpy as np
from loguru import logger

from mayuri.config import settings


class PiperTTS:
    """Wraps Piper TTS for local text-to-speech synthesis."""

    def __init__(self) -> None:
        self._model_path = settings.piper_model_path
        if not self._model_path.exists():
            logger.warning(
                f"Piper model not found at {self._model_path}. "
                "Run `python scripts/install_models.py` to download."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Synthesise *text* and play through default speakers (blocking)."""
        if not text.strip():
            return

        logger.debug(f"TTS: '{text[:60]}…'")
        audio, sample_rate = self._synthesize(text)
        sd.play(audio, samplerate=sample_rate, blocking=True)

    def synthesize_to_file(self, text: str, output_path: Path) -> None:
        """Synthesise *text* and write to a WAV file."""
        audio, sample_rate = self._synthesize(text)
        sf.write(str(output_path), audio, sample_rate)

    # ── Private ───────────────────────────────────────────────────────────────

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Run Piper CLI and return (audio_array, sample_rate)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            "piper",
            "--model", str(self._model_path),
            "--output_file", tmp_path,
        ]

        result = subprocess.run(
            cmd,
            input=text.encode(),
            capture_output=True,
        )

        if result.returncode != 0:
            logger.error(f"Piper failed: {result.stderr.decode()}")
            return np.zeros(1000, dtype=np.float32), 22050

        audio, sample_rate = sf.read(tmp_path, dtype="float32")
        Path(tmp_path).unlink(missing_ok=True)
        return audio, sample_rate
