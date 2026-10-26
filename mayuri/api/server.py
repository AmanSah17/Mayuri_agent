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
import json
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from mayuri.api.routes import chat, memory, status
from mayuri.api.ws.events import EventBus
from mayuri.config import settings

# Thread pool for blocking TTS
_tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Mayuri API",
        description="Local-first AI operating layer — internal API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    # CORS — allow localhost on any port for separate frontend dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(chat.router)
    app.include_router(status.router)
    app.include_router(memory.router)

    # Root redirect to UI
    @app.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        return RedirectResponse(url="/ui/")

    # Serve UI static files
    ui_dir = settings.base_dir / "mayuri" / "ui"
    if ui_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

    # Real-time WebSocket endpoint
    @app.websocket(settings.api_ws_path)
    async def ws_events(websocket: WebSocket) -> None:
        await EventBus.connect(websocket)

        async def _handle_chat(user_text: str, session_id: str) -> None:
            """Run agent + TTS in a background task so the receive loop stays free."""
            try:
                from mayuri.agent.graph import MayuriAgent
                from mayuri.tts.synthesizer import PiperTTS

                await EventBus.emit("state", {"status": "thinking"})
                await EventBus.emit("transcript", {"text": user_text})

                agent = MayuriAgent()
                response_text = await agent.run(user_input=user_text, session_id=session_id)

                await EventBus.emit("state", {"status": "speaking"})
                await EventBus.emit("response", {"text": response_text})

                # Speak the response through the local speaker (non-blocking via thread)
                if response_text.strip():
                    loop = asyncio.get_running_loop()
                    tts = PiperTTS()
                    await loop.run_in_executor(_tts_executor, tts.speak, response_text)

                await EventBus.emit("state", {"status": "idle"})
            except Exception as exc:
                logger.error(f"Chat handler error: {exc}")
                await EventBus.emit("error", {"message": str(exc)})

        try:
            while True:
                msg_text = await websocket.receive_text()
                if not msg_text or msg_text.strip() == "ping":
                    continue

                try:
                    payload = json.loads(msg_text)
                    if payload.get("type") == "chat":
                        user_text = payload.get("text", "").strip()
                        session_id = payload.get("session_id", "default")
                        if user_text:
                            # Fire-and-forget: agent runs freely, receive loop stays unblocked
                            asyncio.create_task(_handle_chat(user_text, session_id))
                except Exception as exc:
                    logger.error(f"WebSocket message parse error: {exc}")
                    await EventBus.emit("error", {"message": str(exc)})
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
