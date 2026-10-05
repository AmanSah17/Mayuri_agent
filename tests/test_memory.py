"""
tests/test_memory.py
─────────────────────
Integration test for the async SQLite memory store.
Uses a temporary in-memory SQLite DB.
"""

import asyncio
import pytest
from unittest.mock import patch

from mayuri.memory.store import MemoryStore


@pytest.mark.asyncio
async def test_save_and_load_messages(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    with patch("mayuri.memory.store.settings") as mock_settings:
        mock_settings.memory_db_path = db_path
        mock_settings.memory_max_history = 10

        store = MemoryStore()
        await store.save_turn("sess1", "Hello Mayuri", "Hello Aman!")

        msgs = await store.load_messages("sess1")
        assert len(msgs) == 2
        assert msgs[0].content == "Hello Mayuri"
        assert msgs[1].content == "Hello Aman!"


@pytest.mark.asyncio
async def test_facts(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    with patch("mayuri.memory.store.settings") as mock_settings:
        mock_settings.memory_db_path = db_path
        mock_settings.memory_max_history = 10

        store = MemoryStore()
        await store.save_fact("user_name", "Aman")
        value = await store.get_fact("user_name")
        assert value == "Aman"

        # Upsert
        await store.save_fact("user_name", "Aman Sharma")
        value = await store.get_fact("user_name")
        assert value == "Aman Sharma"
