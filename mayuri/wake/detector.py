"""
mayuri/wake/detector.py
────────────────────────
Thin wrapper around openWakeWord.

openWakeWord uses ONNX Runtime under the hood, so this is extremely
lightweight — it never loads the LLM or any heavy model.

Design:
  - Feed 80 ms chunks of 16 kHz mono PCM int16 audio.
  - Returns True the moment the wake phrase confidence crosses the threshold.
  - Resets internal state after each detection so back-to-back activations work.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from mayuri.config import settings


class WakeWordDetector:
    """Wraps openWakeWord for the 'Mayuri' trigger phrase."""

    # openWakeWord expects 80 ms frames at 16 kHz → 1280 samples
    CHUNK_SAMPLES: int = 1280
    SAMPLE_RATE: int = 16_000

    def __init__(self) -> None:
        self._model = self._load_model()
        logger.info(f"Wake-word detector ready (phrase='{settings.wake_word}', "
                    f"threshold={settings.wake_word_threshold})")

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_model() -> object:
        """Load openWakeWord model. Returns the oww Model object."""
        try:
            import openwakeword  # type: ignore[import]
            from openwakeword.model import Model  # type: ignore[import]

            # openwakeword ships a pretrained "hey_mycroft" style model.
            # For a custom "Mayuri" wake word you need to train your own ONNX.
            # Point WAKE_WORD to the model path or a built-in keyword.
            model = Model(
                wakeword_models=[settings.wake_word],
                inference_framework="onnx",
            )
            return model
        except ImportError as exc:
            raise RuntimeError(
                "openWakeWord is not installed. Run: pip install openwakeword"
            ) from exc

    # ── Public API ────────────────────────────────────────────────────────────

    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Feed one 80 ms PCM chunk.

        Parameters
        ----------
        audio_chunk:
            np.ndarray of shape (CHUNK_SAMPLES,) dtype int16 or float32.

        Returns
        -------
        bool: True if wake word detected above threshold.
        """
        # openWakeWord expects float32 in [-1, 1]
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0

        prediction = self._model.predict(audio_chunk)

        # prediction is a dict: {model_name: score}
        scores = list(prediction.values())
        if not scores:
            return False

        max_score: float = max(scores)
        detected = max_score >= settings.wake_word_threshold

        if detected:
            logger.debug(f"Wake word detected! score={max_score:.3f}")
            self._model.reset()  # prevent double-fire

        return detected
