"""
mayuri/audio/vad.py
────────────────────
Silero VAD wrapper.

Segments a stream of PCM chunks into utterances by detecting speech onset
and offset. Returns complete utterances (as raw bytes) for the STT layer.

Key parameters (all configurable via Settings):
  vad_threshold      — probability above which a frame is "speech"
  vad_min_silence_ms — ms of silence required to end an utterance
  vad_speech_pad_ms  — ms of padding added before/after speech

Usage:
    vad = SileroVAD()
    for chunk in microphone:
        utterance = vad.process(chunk)
        if utterance is not None:
            text = whisper.transcribe(utterance)
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

from mayuri.config import settings


class SileroVAD:
    """Stateful VAD that emits complete speech utterances."""

    SAMPLE_RATE = 16_000
    CHUNK_SAMPLES = 512   # ~32 ms at 16 kHz — Silero's preferred chunk size

    def __init__(self) -> None:
        self._model, self._utils = self._load_model()
        self._get_speech_timestamps = self._utils[0]
        self._threshold = settings.vad_threshold
        self._min_silence_samples = int(
            settings.vad_min_silence_ms / 1000 * self.SAMPLE_RATE
        )
        self._speech_pad_samples = int(
            settings.vad_speech_pad_ms / 1000 * self.SAMPLE_RATE
        )

        # Rolling state
        self._buffer: list[np.ndarray] = []
        self._speaking = False
        self._silence_count = 0

        logger.info("Silero VAD loaded.")

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_model() -> tuple[object, tuple]:
        model, utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=False,
            onnx=False,
        )
        model.eval()
        return model, utils

    def _is_speech(self, chunk: np.ndarray) -> bool:
        tensor = torch.from_numpy(chunk.astype(np.float32))
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            prob: float = self._model(tensor, self.SAMPLE_RATE).item()
        return prob >= self._threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, chunk: np.ndarray) -> bytes | None:
        """
        Feed one audio chunk.

        Returns
        -------
        bytes | None
            Raw 16-bit PCM bytes of the complete utterance when speech ends,
            otherwise None.
        """
        speech = self._is_speech(chunk)

        if speech:
            self._speaking = True
            self._silence_count = 0
            self._buffer.append(chunk)
        elif self._speaking:
            self._silence_count += len(chunk)
            self._buffer.append(chunk)

            if self._silence_count >= self._min_silence_samples:
                # End of utterance
                utterance = np.concatenate(self._buffer)
                self._buffer = []
                self._speaking = False
                self._silence_count = 0
                logger.debug(f"VAD: utterance captured ({len(utterance)} samples)")
                return utterance.astype(np.int16).tobytes()

        return None

    def reset(self) -> None:
        self._buffer = []
        self._speaking = False
        self._silence_count = 0
