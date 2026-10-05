"""
mayuri/config.py
────────────────
Single source of truth for all runtime configuration.
All values are loaded from .env via pydantic-settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Project root ──────────────────────────────────────────────────────────
    base_dir: Path = Path(__file__).resolve().parent.parent

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"

    # ── Audio ─────────────────────────────────────────────────────────────────
    audio_device_index: int | None = None
    audio_sample_rate: int = 16_000
    audio_channels: int = 1

    # ── Wake word ─────────────────────────────────────────────────────────────
    wake_word: str = "mayuri"
    wake_word_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # ── STT ───────────────────────────────────────────────────────────────────
    whisper_model_size: Literal["tiny", "base", "small", "medium", "large-v3"] = "medium"
    whisper_device: Literal["cuda", "cpu", "auto"] = "auto"
    whisper_compute_type: Literal["float16", "int8", "float32"] = "float16"

    # ── TTS ───────────────────────────────────────────────────────────────────
    piper_model_path: Path = Path("models/tts/en_US-lessac-medium.onnx")
    piper_speaker_id: int = 0

    # ── VAD ───────────────────────────────────────────────────────────────────
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    vad_min_silence_ms: int = 700
    vad_speech_pad_ms: int = 300

    # ── FastAPI ───────────────────────────────────────────────────────────────
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    api_ws_path: str = "/ws/events"

    # ── PostgreSQL (long-term memory) ─────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "mayuri"
    postgres_user: str = "postgres"
    postgres_password: str = "Amansah@1717"
    postgres_url: str = "postgresql+asyncpg://postgres:Amansah%401717@localhost:5432/mayuri"
    postgres_sync_url: str = "postgresql://postgres:Amansah%401717@localhost:5432/mayuri"

    # ── SQLite (fast session cache) ───────────────────────────────────────────
    memory_db_path: Path = Path("data/mayuri.db")
    memory_max_history: int = 50

    # ── LangSmith ─────────────────────────────────────────────────────────────
    langchain_tracing_v2: str = "true"
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "Mayuri"

    # ── Web search ────────────────────────────────────────────────────────────
    tavily_api_key: str = ""
    exa_api_key: str = ""
    serper_api_key: str = ""        # kept for compatibility
    brave_api_key: str = ""

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: Path = Path("logs/mayuri.log")

    # ── Computed ──────────────────────────────────────────────────────────────
    @property
    def api_base_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"

    @property
    def api_ws_url(self) -> str:
        return f"ws://{self.api_host}:{self.api_port}{self.api_ws_path}"

    def apply_langsmith_env(self) -> None:
        """Inject LangSmith vars into os.environ so LangChain picks them up."""
        os.environ.setdefault("LANGCHAIN_TRACING_V2", self.langchain_tracing_v2)
        os.environ.setdefault("LANGCHAIN_ENDPOINT", self.langchain_endpoint)
        os.environ.setdefault("LANGCHAIN_API_KEY", self.langchain_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", self.langchain_project)


# Global singleton
settings = Settings()
settings.apply_langsmith_env()
