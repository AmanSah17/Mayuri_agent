"""
mayuri/main.py
──────────────
Top-level launcher. Starts every Mayuri service in the right order:

  1. FastAPI backend  (handles HTTP + WebSocket)
  2. Wake-word detector  (always-on, tiny)
  3. Audio pipeline  (VAD → STT → agent → TTS)

Each service runs in its own asyncio Task so a crash in one doesn't
bring down the others. Structured concurrency via anyio / asyncio.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

from loguru import logger

from mayuri.config import settings
from mayuri.logging_setup import configure_logging


import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Mayuri Local-First AI Assistant")
    parser.add_argument(
        "--backend", "--api-only",
        dest="backend_only",
        action="store_true",
        help="Run only the FastAPI backend server (HTTP, WebSocket, LangGraph Agent, Memory) without microphone/audio",
    )
    parser.add_argument(
        "--audio-only",
        dest="audio_only",
        action="store_true",
        help="Run only the microphone listener, wake word, and audio pipeline",
    )
    args = parser.parse_args()

    configure_logging()
    mode_str = "backend only" if args.backend_only else ("audio only" if args.audio_only else "full stack")
    logger.info(f"🌸 Mayuri is starting up [{mode_str}] …")

    # Ensure required directories exist
    for d in [settings.memory_db_path.parent, settings.log_file.parent, Path("models")]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(_run(backend_only=args.backend_only, audio_only=args.audio_only))
    except KeyboardInterrupt:
        logger.info("Mayuri shut down cleanly.")


async def _run(backend_only: bool = False, audio_only: bool = False) -> None:
    stop_event = asyncio.Event()

    def _on_signal(*_: object) -> None:
        logger.info("Shutdown signal received.")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            signal.signal(sig, _on_signal)  # type: ignore[arg-type]

    tasks: list[asyncio.Task] = []

    if not audio_only:
        from mayuri.api.server import start_api_server
        tasks.append(asyncio.create_task(start_api_server(), name="api"))

    if not backend_only:
        from mayuri.wake.service import WakeWordService
        from mayuri.audio.pipeline import AudioPipeline
        wake_service = WakeWordService()
        audio_pipeline = AudioPipeline()
        tasks.append(asyncio.create_task(wake_service.run(stop_event), name="wake"))
        tasks.append(asyncio.create_task(audio_pipeline.run(wake_service, stop_event), name="audio"))

    await stop_event.wait()

    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("All services stopped.")


if __name__ == "__main__":
    main()
