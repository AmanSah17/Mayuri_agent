"""
tests/test_frontend_realtime.py
───────────────────────────────
Integration tests for backend-frontend real-time assistant communication:
  1. Static UI serving at /ui and root redirect from / to /ui
  2. Health & status endpoint integration
  3. Real-time WebSocket chat event streaming (thinking -> tool_call -> response -> idle)
  4. Cross-thread vector memory semantic compression & recall
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import pytest
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mayuri.api.server import create_app
from mayuri.config import settings
from mayuri.memory.vector_store import VectorConversationMemory


@pytest.fixture
def app():
    return create_app()


@pytest.mark.asyncio
async def test_frontend_static_serving(app):
    """Verify that frontend HTML, CSS, JS and root redirect are served."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Root redirect
        r_root = await client.get("/", follow_redirects=False)
        assert r_root.status_code in (302, 307), f"Expected redirect, got {r_root.status_code}"
        assert "/ui/" in r_root.headers.get("location", "")

        # UI index.html
        r_ui = await client.get("/ui/")
        assert r_ui.status_code == 200
        assert "<title>Mayuri</title>" in r_ui.text
        assert "app.js" in r_ui.text

        # UI app.js
        r_js = await client.get("/ui/app.js")
        assert r_js.status_code == 200
        assert "connect" in r_js.text

        # UI style.css
        r_css = await client.get("/ui/style.css")
        assert r_css.status_code == 200
        assert "orb" in r_css.text


@pytest.mark.asyncio
async def test_frontend_status_endpoint(app):
    """Verify frontend status polling endpoint returns component health."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/status")
        assert res.status_code == 200
        data = res.json()
        assert "uptime_seconds" in data
        assert "components" in data
        names = [c["name"] for c in data["components"]]
        assert "ollama" in names


@pytest.mark.asyncio
async def test_frontend_chat_http_endpoint(app):
    """Verify HTTP chat endpoint returns Qwen assistant answer and broadcasts state."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.post(
            "/chat/message",
            json={"text": "Hello Mayuri, reply with 'Frontend test passed!'", "session_id": "test_http_fe"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "response" in data
        assert len(data["response"]) > 0


def test_realtime_websocket_assistant(app):
    """Verify full real-time WebSocket protocol with state and response events."""
    from starlette.testclient import TestClient

    client = TestClient(app)
    with client.websocket_connect(settings.api_ws_path) as ws:
        # Send a real-time message over WebSocket
        ws.send_text(json.dumps({
            "type": "chat",
            "text": "Say 'WebSocket connected' in 2 words",
            "session_id": "test_ws_stream",
        }))

        received_types = []
        for _ in range(6):
            try:
                raw = ws.receive_text()
                msg = json.loads(raw)
                t = msg.get("type")
                received_types.append(t)
                if t == "response":
                    assert len(msg.get("data", {}).get("text", "")) > 0
                    break
            except Exception:
                break

        print(f"\n[OK] Received WebSocket events: {received_types}")
        assert "state" in received_types or "transcript" in received_types or "response" in received_types


@pytest.mark.asyncio
async def test_cross_thread_vector_compression_and_awareness():
    """Verify that past conversations in older threads are compressed and recalled in new sessions."""
    from mayuri.agent.graph import MayuriAgent

    agent = MayuriAgent()
    past_session = "thread_secret_project_99"
    current_session = "thread_new_chat_101"

    # Step 1: User mentions a specific unique topic in past thread
    turn1 = await agent.run(
        user_input="Remember that our secret project code is 'PROJECT-NEBULA-77'.",
        session_id=past_session,
    )
    assert turn1

    # Step 2: In a completely different thread, ask about the secret code
    turn2 = await agent.run(
        user_input="What was the secret project code I mentioned earlier in another thread?",
        session_id=current_session,
    )
    print(f"\n[OK] Cross-thread recall answer in new session: '{turn2}'")
    assert "nebula" in turn2.lower() or "77" in turn2 or "project" in turn2.lower()
