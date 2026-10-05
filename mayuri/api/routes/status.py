"""
mayuri/api/routes/status.py
────────────────────────────
System status and health endpoints.

GET /status          — Current Mayuri state and component health
GET /status/devices  — List available audio input devices
"""

from __future__ import annotations

import platform
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/status", tags=["status"])

_start_time = datetime.utcnow()


class ComponentStatus(BaseModel):
    name: str
    status: str   # "ok" | "error" | "not_loaded"
    detail: str = ""


class StatusResponse(BaseModel):
    uptime_seconds: float
    platform: str
    components: list[ComponentStatus]


@router.get("", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Return health status of all Mayuri components."""
    uptime = (datetime.utcnow() - _start_time).total_seconds()
    components = []

    # Check Ollama
    try:
        import httpx
        from mayuri.config import settings
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0)
        components.append(ComponentStatus(name="ollama", status="ok", detail=str(r.status_code)))
    except Exception as exc:
        components.append(ComponentStatus(name="ollama", status="error", detail=str(exc)))

    # Check faster-whisper
    try:
        import faster_whisper  # noqa: F401
        components.append(ComponentStatus(name="whisper", status="ok"))
    except ImportError:
        components.append(ComponentStatus(name="whisper", status="not_loaded"))

    # Check sounddevice
    try:
        import sounddevice as sd
        sd.query_devices()
        components.append(ComponentStatus(name="audio", status="ok"))
    except Exception as exc:
        components.append(ComponentStatus(name="audio", status="error", detail=str(exc)))

    return StatusResponse(
        uptime_seconds=uptime,
        platform=platform.version(),
        components=components,
    )


@router.get("/devices")
async def list_audio_devices() -> dict:
    """Return all available audio input devices."""
    import sounddevice as sd
    devices = sd.query_devices()
    result = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            result.append({"index": i, "name": d["name"], "channels": d["max_input_channels"]})
    return {"devices": result}
