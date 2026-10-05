"""
mayuri/api/server.py
─────────────────────
FastAPI application factory and server launcher.

Exposes:
  HTTP  http://127.0.0.1:8765/
  WS    ws://127.0.0.1:8765/ws/events

The UI connects to the WebSocket for real-time event streaming and uses
the HTTP endpoints for text chat and memory management.
"""

from __future__ import annotations

import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from mayuri.api.routes import chat, memory, status
from mayuri.api.ws.events import EventBus
from mayuri.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="Mayuri API",
        description="Local-first AI operating layer — internal API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    # CORS — only localhost
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[f"http://{settings.api_host}:{settings.api_port}", "http://localhost:*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(chat.router)
    app.include_router(status.router)
    app.include_router(memory.router)

    # Serve UI static files
    ui_dir = settings.base_dir / "mayuri" / "ui"
    if ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    # WebSocket endpoint
    @app.websocket(settings.api_ws_path)
    async def ws_events(websocket: WebSocket) -> None:
        await EventBus.connect(websocket)
        try:
            while True:
                # Keep connection alive; client sends pings
                await websocket.receive_text()
        except WebSocketDisconnect:
            await EventBus.disconnect(websocket)

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info(f"🚀 Mayuri API running at {settings.api_base_url}")
        logger.info(f"   WebSocket: {settings.api_ws_url}")
        logger.info(f"   Docs:      {settings.api_base_url}/docs")

    return app


async def start_api_server() -> None:
    """Launch uvicorn programmatically (called from main.py)."""
    app = create_app()
    config = uvicorn.Config(
        app=app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="warning",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    await server.serve()


def main() -> None:
    """Entry point for `mayuri-api` CLI command."""
    import asyncio
    asyncio.run(start_api_server())
