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

from fastapi import APIRouter
from pydantic import BaseModel

from mayuri.agent.graph import MayuriAgent

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    text: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


@router.post("/message", response_model=ChatResponse)
async def chat_message(req: ChatRequest) -> ChatResponse:
    """Process a text message through the Mayuri agent."""
    agent = MayuriAgent()
    response = await agent.run(user_input=req.text, session_id=req.session_id)
    return ChatResponse(response=response, session_id=req.session_id)
