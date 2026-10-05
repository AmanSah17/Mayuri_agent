"""
mayuri/api/ws/events.py
────────────────────────
WebSocket event bus.

Provides a simple publish/subscribe hub so any part of Mayuri can
push events to all connected UI clients without knowing about them.

Events are JSON objects with at minimum:
  { "type": "<event_type>", "data": {...} }

Known event types:
  "state"       — Mayuri's current state: idle | listening | thinking | speaking
  "transcript"  — STT result: { "text": "..." }
  "response"    — Agent response: { "text": "..." }
  "tool_call"   — Tool being invoked: { "tool": "...", "args": {...} }
  "error"       — Error occurred: { "message": "..." }
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import WebSocket
from loguru import logger


class _EventBus:
    """Singleton WebSocket broadcast hub."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        logger.info(f"UI connected. Total clients: {len(self._clients)}")

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        logger.info(f"UI disconnected. Total clients: {len(self._clients)}")

    async def emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Broadcast an event to all connected WebSocket clients."""
        payload = json.dumps({"type": event_type, "data": data or {}})
        dead: list[WebSocket] = []

        async with self._lock:
            clients = list(self._clients)

        for ws in clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)


# Singleton
EventBus = _EventBus()
