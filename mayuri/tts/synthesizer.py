"""
mayuri/tts/synthesizer.py
──────────────────────────
Piper TTS wrapper with voice smoothing and pitch tuning.

Piper is a fast, local neural TTS engine (ONNX-based).
Tuned for:
  - Ultra-smooth phoneme transitions (low noise_scale)
  - Lower, warmer pitch (post-synthesis pitch shifting)
  - Sibilance softening filter
"""

from __future__ import annotations

import io
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger

from mayuri.config import settings


class PiperTTS:
    """Wraps Piper TTS for local text-to-speech synthesis with acoustic tuning."""

    _PIPER_EXE: Path | None = None

    @classmethod
    def _find_piper_exe(cls) -> Path | None:
        if cls._PIPER_EXE is not None:
            return cls._PIPER_EXE

        scripts_dir = Path(sys.executable).parent
        candidate = scripts_dir / "piper.exe"
        if candidate.exists():
            cls._PIPER_EXE = candidate
            return candidate

        found = shutil.which("piper")
        if found:
            cls._PIPER_EXE = Path(found)
            return cls._PIPER_EXE

        return None

    def __init__(self) -> None:
        self._model_path = settings.piper_model_path
        self._voice = None
        self._syn_config = None

        if not self._model_path.exists():
            logger.warning(
                f"Piper model not found at {self._model_path}. "
                "Run `python scripts/install_models.py` to download."
            )
        else:
            self._init_piper_voice()

        piper_exe = self._find_piper_exe()
        if piper_exe:
            logger.debug(f"Piper binary fallback: {piper_exe}")

    def _init_piper_voice(self) -> None:
        """Pre-load PiperVoice ONNX model and configure synthesis parameters for smoother voice."""
        try:
            import piper
            from piper.config import SynthesisConfig

            logger.info(f"Loading Piper voice model from {self._model_path} …")
            config_path = str(self._model_path) + ".json"
            self._voice = piper.PiperVoice.load(
                str(self._model_path),
                config_path=config_path if Path(config_path).exists() else None,
                use_cuda=False,
            )
            # Tuning for smoother, warmer voice:
            # - noise_scale=0.333: reduces phoneme jitter and breathiness for a silkier, steady tone
            # - noise_w_scale=0.60: stabilizes phoneme duration
            # - length_scale=1.08: relaxed, calm pacing
            self._syn_config = SynthesisConfig(
                noise_scale=0.333,
                noise_w_scale=0.60,
                length_scale=1.08,
            )
            logger.info("Piper voice model ready (smooth acoustic profile loaded).")
        except Exception as exc:
            logger.warning(f"Could not initialize direct Python PiperVoice: {exc}")
            self._voice = None

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Synthesise *text*, apply audio smoothing/pitch reduction, and play."""
        if not text.strip():
            return

        logger.debug(f"TTS: '{text[:60]}…'")
        try:
            audio, sample_rate = self._synthesize(text)
            sd.play(audio, samplerate=sample_rate, blocking=True)
        except Exception as exc:
            logger.error(f"TTS speak failed: {exc}")

    def synthesize_to_file(self, text: str, output_path: Path) -> None:
        """Synthesise *text* and write to a WAV file."""
        audio, sample_rate = self._synthesize(text)
        sf.write(str(output_path), audio, sample_rate)

    # ── Private ───────────────────────────────────────────────────────────────

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize audio and apply post-processing for smoother tone and lower pitch."""
        raw_audio: np.ndarray | None = None
        sample_rate: int = 22050

        # Strategy 1: Direct Python PiperVoice (cached, fast)
        if self._voice is not None:
            try:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wav_file:
                    self._voice.synthesize_wav(text, wav_file, syn_config=self._syn_config)
                buf.seek(0)
                raw_audio, sample_rate = sf.read(buf, dtype="float32")
            except Exception as exc:
                logger.warning(f"Direct Python synthesis failed ({exc}), trying subprocess…")

        # Strategy 2: piper.exe subprocess fallback
        if raw_audio is None:
            raw_audio, sample_rate = self._synthesize_via_subprocess(text)

        # Apply voice smoothing and pitch reduction
        processed_audio = self._post_process_audio(raw_audio, sample_rate)
        return processed_audio, sample_rate

    def _post_process_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Lower pitch slightly and apply gentle warm low-pass filter to soften highs/harshness."""
        if len(audio) < 100:
            return audio

        out = audio.copy()

        # 1. Lower pitch by -2.0 semitones for a deeper, calmer, less high-pitched voice
        try:
            import torch
            import torchaudio.functional as F

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tensor = torch.from_numpy(out).unsqueeze(0).to(device)
            shifted = F.pitch_shift(tensor, sample_rate, n_steps=-2.0)
            out = shifted.squeeze(0).cpu().numpy()
        except Exception as exc:
            logger.debug(f"Pitch shift bypassed ({exc})")

        # 2. Gentle warm filter (lowpass at 7500 Hz) to eliminate sharp sibilance/roughness
        try:
            from scipy.signal import butter, sosfilt
            sos = butter(2, 7500, btype="low", fs=sample_rate, output="sos")
            out = sosfilt(sos, out).astype(np.float32)
        except Exception:
            pass

        return out

    def _synthesize_via_subprocess(self, text: str) -> tuple[np.ndarray, int]:
        """Run piper.exe via subprocess as fallback."""
        import subprocess
        import tempfile

        piper_exe = self._find_piper_exe()
        if piper_exe is None:
            logger.error("No piper binary found. TTS unavailable.")
            return np.zeros(1000, dtype=np.float32), 22050

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            str(piper_exe),
            "--model", str(self._model_path),
            "--output_file", tmp_path,
            "--length_scale", "1.08",
            "--noise_scale", "0.333",
            "--noise_w", "0.6",
        ]

        result = subprocess.run(cmd, input=text.encode(), capture_output=True)

        if result.returncode != 0:
            logger.error(f"Piper subprocess failed: {result.stderr.decode()}")
            return np.zeros(1000, dtype=np.float32), 22050

        audio, sample_rate = sf.read(tmp_path, dtype="float32")
        Path(tmp_path).unlink(missing_ok=True)
        return audio, sample_rate
