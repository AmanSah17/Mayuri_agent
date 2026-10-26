"""
mayuri/api/routes/chat.py
──────────────────────────
HTTP chat endpoint — text-only alternative to voice.

POST /chat/message
  Body: { "text": "...", "session_id": "default" }
  Returns: { "response": "..." }

Useful for:
  - UI text input box
  - Testing without microphone
  - Programmatic integration
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter
from pydantic import BaseModel

from mayuri.agent.graph import MayuriAgent
from mayuri.api.ws.events import EventBus

_tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-http")

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    text: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


@router.post("/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest) -> ChatResponse:
    """Process a text message through the Mayuri agent and emit real-time events."""
    await EventBus.emit("state", {"status": "thinking"})
    await EventBus.emit("transcript", {"text": req.text})

    agent = MayuriAgent()
    response = await agent.run(user_input=req.text, session_id=req.session_id)

    await EventBus.emit("state", {"status": "speaking"})
    await EventBus.emit("response", {"text": response})

    # Speak the response aloud (non-blocking)
    if response.strip():
        from mayuri.tts.synthesizer import PiperTTS
        loop = asyncio.get_running_loop()
        tts = PiperTTS()
        await loop.run_in_executor(_tts_executor, tts.speak, response)

    await EventBus.emit("state", {"status": "idle"})

    return ChatResponse(response=response, session_id=req.session_id)
