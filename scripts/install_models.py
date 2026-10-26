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
    msg = msg.replace("✓", "[OK]").replace("✗", "[FAIL]")
    try:
        print(f"  {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"  {msg.encode('ascii', 'replace').decode('ascii')}", flush=True)


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
        try:
            from mayuri.config import settings
            model_size = settings.whisper_model_size
        except Exception:
            model_size = "base"
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
        torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False, onnx=False, trust_repo=True)
        log("✓ Silero VAD cached.")
    except ImportError:
        log("✗ torch not installed. Run: pip install torch")
    except Exception as exc:
        log(f"✗ Failed: {exc}")


# ── 5. Ollama LLM (Qwen) ──────────────────────────────────────────────────────

def verify_ollama_model() -> None:
    print("\n[5/5] Ollama LLM (Qwen)")
    try:
        import json
        import urllib.request
        from mayuri.config import settings

        url = f"{settings.ollama_base_url}/api/tags"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            models = [m.get("name", "") for m in data.get("models", [])]
            if any(settings.ollama_model in m for m in models):
                log(f"✓ Ollama model '{settings.ollama_model}' is available.")
            else:
                log(f"✗ Model '{settings.ollama_model}' not found. Run: ollama pull {settings.ollama_model}")
    except Exception as exc:
        log(f"✗ Ollama check failed: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Mayuri — Model Installer")
    print("=" * 50)

    install_openwakeword()
    install_whisper()
    install_piper()
    install_silero()
    verify_ollama_model()

    print("\n" + "=" * 50)
    print("  Done! Run `mayuri` to start.")
    print("=" * 50)
