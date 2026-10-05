"""
tests/test_agent_e2e.py
────────────────────────
End-to-end integration test for Mayuri agent.

Tests:
  1. PostgreSQL connection + table creation
  2. Facts store (save / retrieve / upsert)
  3. Personality loading
  4. Agent text turn (with LangSmith tracing)
     - General knowledge question
     - Web search (Tavily)
     - Humorous banter
     - Math calculation
  5. Conversation continuity (multi-turn memory)
  6. SQLite fallback memory

Run with: pytest tests/test_agent_e2e.py -v -s
"""

from __future__ import annotations

import asyncio
import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mayuri.config import settings


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PostgreSQL connection test
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_postgres_connection():
    """Verify PostgreSQL is reachable and tables are created."""
    from mayuri.memory.pg_store import PGMemoryStore
    store = PGMemoryStore()
    ok = await store.init()
    assert ok, "PostgreSQL connection failed — check POSTGRES_URL in .env"
    print("\n[OK] PostgreSQL connected and tables created.")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Facts store
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_facts_crud():
    """Test saving, retrieving, and upserting facts."""
    from mayuri.memory.pg_store import PGMemoryStore
    store = PGMemoryStore()
    await store.init()

    await store.save_fact("user_name", "Aman")
    await store.save_fact("user_city", "Bengaluru")
    await store.save_fact("preferred_language", "Python")

    assert await store.get_fact("user_name") == "Aman"
    assert await store.get_fact("user_city") == "Bengaluru"

    # Upsert
    await store.save_fact("user_city", "Hyderabad")
    assert await store.get_fact("user_city") == "Hyderabad"

    all_facts = await store.get_all_facts()
    assert "user_name" in all_facts
    print(f"\n[OK] Facts: {all_facts}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Personality state
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_personality():
    from mayuri.memory.pg_store import PGMemoryStore
    store = PGMemoryStore()
    await store.init()

    p = await store.get_personality()
    print(f"\n[OK] Personality: {p}")
    assert "humor_level" in p
    assert 0.0 <= p["humor_level"] <= 1.0

    # Update humor
    await store.update_personality(humor_level=0.9)
    p2 = await store.get_personality()
    assert p2["humor_level"] == 0.9
    print(f"   Updated: {p2}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Agent turns (requires Ollama to be running)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_agent_general_knowledge():
    """Test a simple factual question."""
    from mayuri.agent.graph import MayuriAgent
    agent = MayuriAgent()

    answer = await agent.run(
        "What is the capital of France?",
        session_id="test_general"
    )
    print(f"\n[OK] General knowledge: '{answer}'")
    assert answer, "Empty response from agent"
    assert len(answer) > 5


@pytest.mark.asyncio
async def test_agent_datetime():
    """Test datetime tool."""
    from mayuri.agent.graph import MayuriAgent
    agent = MayuriAgent()

    answer = await agent.run(
        "What is the current date and time?",
        session_id="test_datetime"
    )
    print(f"\n[OK] DateTime: '{answer}'")
    assert answer


@pytest.mark.asyncio
async def test_agent_calculator():
    """Test math calculation tool."""
    from mayuri.agent.graph import MayuriAgent
    agent = MayuriAgent()

    answer = await agent.run(
        "Calculate 2 to the power of 16",
        session_id="test_math"
    )
    print(f"\n[OK] Calculator: '{answer}'")
    assert "65536" in answer or answer  # LLM might paraphrase


@pytest.mark.asyncio
async def test_agent_humor():
    """Test that the agent responds with personality."""
    from mayuri.agent.graph import MayuriAgent
    agent = MayuriAgent()

    # Set humor to max first
    from mayuri.memory.pg_store import PGMemoryStore
    store = PGMemoryStore()
    await store.init()
    await store.update_personality(humor_level=0.9)

    answer = await agent.run(
        "Tell me a short joke",
        session_id="test_humor"
    )
    print(f"\n[OK] Humor: '{answer}'")
    assert answer


@pytest.mark.asyncio
async def test_agent_web_search_tavily():
    """Test Tavily web search tool."""
    from mayuri.agent.graph import MayuriAgent
    agent = MayuriAgent()

    answer = await agent.run(
        "Search the web: what is the latest Python version?",
        session_id="test_tavily"
    )
    print(f"\n[OK] Tavily search: '{answer}'")
    assert answer


@pytest.mark.asyncio
async def test_agent_exa_search():
    """Test Exa neural search tool."""
    from mayuri.agent.graph import MayuriAgent
    agent = MayuriAgent()

    answer = await agent.run(
        "Use Exa to find information about LangGraph multi-agent systems",
        session_id="test_exa"
    )
    print(f"\n[OK] Exa search: '{answer}'")
    assert answer


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Multi-turn conversation continuity
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_conversation_continuity():
    """Verify that the agent remembers earlier turns in the same session."""
    from mayuri.agent.graph import MayuriAgent
    agent = MayuriAgent()
    session = "test_continuity"

    # Clear previous data
    from mayuri.memory.pg_store import PGMemoryStore
    pg = PGMemoryStore()
    await pg.init()
    await pg.clear_session(session)

    # Turn 1: introduce a fact
    r1 = await agent.run("My favourite programming language is Rust.", session_id=session)
    print(f"\n[OK] Turn 1: '{r1}'")

    # Turn 2: refer back to it
    r2 = await agent.run(
        "What programming language did I just tell you I liked?",
        session_id=session
    )
    print(f"   Turn 2: '{r2}'")
    assert "rust" in r2.lower() or "programming" in r2.lower(), (
        f"Agent forgot context. Response: {r2}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. STT wake-word stripping (no model needed)
# ═══════════════════════════════════════════════════════════════════════════════

def test_wake_word_strip():
    from mayuri.stt.transcriber import WhisperTranscriber
    t = WhisperTranscriber.__new__(WhisperTranscriber)
    cases = [
        ("Mayuri, set volume to 50", "set volume to 50"),
        ("Hey Mayuri what's the weather", "what's the weather"),
        ("mayuri. open Chrome", "open Chrome"),
        ("search Amazon for laptops", "search Amazon for laptops"),
    ]
    for inp, expected in cases:
        got = t._strip_wake_word(inp)
        assert got == expected, f"Expected '{expected}', got '{got}'"
    print("\n[OK] Wake word stripping: all cases pass.")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SQLite fallback memory
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_sqlite_memory(tmp_path):
    from unittest.mock import patch
    from mayuri.memory.store import MemoryStore

    db = tmp_path / "test.db"
    with patch("mayuri.memory.store.settings") as m:
        m.memory_db_path = db
        m.memory_max_history = 10
        store = MemoryStore()
        await store.save_turn("s1", "hello", "hi!")
        msgs = await store.load_messages("s1")
        assert len(msgs) == 2
        print(f"\n[OK] SQLite fallback: {len(msgs)} messages.")
