"""
scripts/install_models.py
──────────────────────────
Download all required models:
  1. openWakeWord pretrained ONNX models
  2. faster-whisper Whisper model
  3. Piper TTS voice model
  4. Silero VAD (via torch.hub)

Run once before first launch: python scripts/install_models.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
TTS_DIR = MODELS_DIR / "tts"
WAKE_DIR = MODELS_DIR / "wake"

MODELS_DIR.mkdir(exist_ok=True)
TTS_DIR.mkdir(exist_ok=True)
WAKE_DIR.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


# ── 1. openWakeWord ───────────────────────────────────────────────────────────

def install_openwakeword() -> None:
    print("\n[1/4] openWakeWord models")
    try:
        import openwakeword  # type: ignore[import]
        openwakeword.utils.download_models()
        log("✓ openWakeWord models downloaded.")
    except ImportError:
        log("✗ openwakeword not installed. Run: pip install openwakeword")
    except Exception as exc:
        log(f"✗ Failed: {exc}")


# ── 2. faster-whisper ─────────────────────────────────────────────────────────

def install_whisper() -> None:
    print("\n[2/4] faster-whisper model")
    try:
        from faster_whisper import WhisperModel  # type: ignore[import]
        model_size = "medium"
        log(f"Downloading Whisper '{model_size}' (this may take a few minutes)…")
        WhisperModel(model_size, device="cpu", compute_type="int8")
        log(f"✓ Whisper '{model_size}' cached.")
    except ImportError:
        log("✗ faster-whisper not installed. Run: pip install faster-whisper")
    except Exception as exc:
        log(f"✗ Failed: {exc}")


# ── 3. Piper TTS ──────────────────────────────────────────────────────────────

PIPER_MODEL_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
)
PIPER_CONFIG_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
)


def install_piper() -> None:
    print("\n[3/4] Piper TTS model")
    for url in [PIPER_MODEL_URL, PIPER_CONFIG_URL]:
        fname = TTS_DIR / url.split("/")[-1]
        if fname.exists():
            log(f"✓ Already exists: {fname.name}")
            continue
        log(f"Downloading {fname.name}…")
        try:
            urllib.request.urlretrieve(url, fname)
            log(f"✓ {fname.name}")
        except Exception as exc:
            log(f"✗ Failed: {exc}")


# ── 4. Silero VAD ─────────────────────────────────────────────────────────────

def install_silero() -> None:
    print("\n[4/4] Silero VAD")
    try:
        import torch  # type: ignore[import]
        log("Downloading Silero VAD via torch.hub…")
        torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False, onnx=False)
        log("✓ Silero VAD cached.")
    except ImportError:
        log("✗ torch not installed. Run: pip install torch")
    except Exception as exc:
        log(f"✗ Failed: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Mayuri — Model Installer")
    print("=" * 50)

    install_openwakeword()
    install_whisper()
    install_piper()
    install_silero()

    print("\n" + "=" * 50)
    print("  Done! Run `mayuri` to start.")
    print("=" * 50)
