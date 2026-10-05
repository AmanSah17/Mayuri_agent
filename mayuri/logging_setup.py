"""
mayuri/logging_setup.py
────────────────────────
Configure loguru for file + stderr output.
"""

from __future__ import annotations

import sys

from loguru import logger

from mayuri.config import settings


def configure_logging() -> None:
    logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )

    # Stderr (rich colours)
    logger.add(sys.stderr, level=settings.log_level, format=fmt, colorize=True)

    # Rotating file
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(settings.log_file),
        level="DEBUG",
        format=fmt,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        enqueue=True,   # thread-safe
    )
