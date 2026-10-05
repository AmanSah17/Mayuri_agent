"""
mayuri/api/routes/memory.py
────────────────────────────
Memory management endpoints.

GET  /memory/{session_id}           — Get conversation history
DELETE /memory/{session_id}         — Clear session history
GET  /memory/facts                  — List all stored facts
PUT  /memory/facts/{key}            — Set a fact
DELETE /memory/facts/{key}          — Delete a fact
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from mayuri.memory.store import MemoryStore

router = APIRouter(prefix="/memory", tags=["memory"])


class FactRequest(BaseModel):
    value: str


@router.get("/{session_id}")
async def get_history(session_id: str) -> dict:
    """Return conversation history for a session."""
    store = MemoryStore()
    messages = await store.load_messages(session_id)
    return {
        "session_id": session_id,
        "messages": [
            {"role": type(m).__name__.lower().replace("message", ""), "content": m.content}
            for m in messages
        ],
    }


@router.delete("/{session_id}")
async def clear_history(session_id: str) -> dict:
    """Clear all messages in a session."""
    store = MemoryStore()
    await store.clear_session(session_id)
    return {"cleared": session_id}


@router.put("/facts/{key}")
async def set_fact(key: str, req: FactRequest) -> dict:
    """Store or update a long-term fact."""
    store = MemoryStore()
    await store.save_fact(key, req.value)
    return {"key": key, "value": req.value}


@router.get("/facts/{key}")
async def get_fact(key: str) -> dict:
    """Retrieve a stored fact."""
    store = MemoryStore()
    value = await store.get_fact(key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"Fact '{key}' not found")
    return {"key": key, "value": value}
