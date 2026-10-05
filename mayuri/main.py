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


def main() -> None:
    configure_logging()
    logger.info("🌸 Mayuri is starting up …")

    # Ensure required directories exist
    for d in [settings.memory_db_path.parent, settings.log_file.parent, Path("models")]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("Mayuri shut down cleanly.")


async def _run() -> None:
    # Lazy imports so each service only loads what it needs
    from mayuri.api.server import start_api_server
    from mayuri.wake.service import WakeWordService
    from mayuri.audio.pipeline import AudioPipeline

    stop_event = asyncio.Event()

    def _on_signal(*_: object) -> None:
        logger.info("Shutdown signal received.")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:
            # Windows does not support add_signal_handler for all signals
            signal.signal(sig, _on_signal)  # type: ignore[arg-type]

    api_task = asyncio.create_task(start_api_server(), name="api")
    wake_service = WakeWordService()
    audio_pipeline = AudioPipeline()

    wake_task = asyncio.create_task(wake_service.run(stop_event), name="wake")
    audio_task = asyncio.create_task(audio_pipeline.run(wake_service, stop_event), name="audio")

    await stop_event.wait()

    for task in (api_task, wake_task, audio_task):
        task.cancel()

    await asyncio.gather(api_task, wake_task, audio_task, return_exceptions=True)
    logger.info("All services stopped.")


if __name__ == "__main__":
    main()
