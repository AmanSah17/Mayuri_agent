"""
mayuri/memory/pg_store.py
──────────────────────────
PostgreSQL-backed long-term memory store.

Responsibilities:
  1. Persist full conversation history across restarts
  2. Store and retrieve user facts (name, preferences, context)
  3. Track Mayuri's personality/humor state
  4. Provide context window slicing (most recent N tokens)

Architecture:
  - Uses asyncpg via SQLAlchemy async engine
  - Auto-creates tables on first run (no migration needed for dev)
  - Falls back gracefully if PostgreSQL is unavailable (logs warning)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from loguru import logger
from sqlalchemy import select, delete, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert

from mayuri.config import settings
from mayuri.memory.pg_models import (
    PGBase, Conversation, PGMessage, Fact, PersonalityState
)


class PGMemoryStore:
    """
    Async PostgreSQL long-term memory.

    Usage:
        store = PGMemoryStore()
        await store.init()
        await store.save_turn(session_id, "hello", "hi there!")
        msgs = await store.load_messages(session_id)
        facts = await store.get_all_facts()
    """

    def __init__(self) -> None:
        self._engine = create_async_engine(
            settings.postgres_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._ready = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> bool:
        """Create tables if they don't exist. Returns True if successful."""
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(PGBase.metadata.create_all)
                # Ensure default personality row exists
                await conn.execute(
                    text(
                        "INSERT INTO personality_state (id, humor_level, verbosity, formality) "
                        "VALUES (1, 0.7, 0.5, 0.3) "
                        "ON CONFLICT (id) DO NOTHING"
                    )
                )
            self._ready = True
            logger.info("✅ PostgreSQL memory store initialised.")
            return True
        except Exception as exc:
            logger.warning(f"⚠️  PostgreSQL unavailable: {exc}. Falling back to SQLite only.")
            self._ready = False
            return False

    # ── Conversation management ───────────────────────────────────────────────

    async def ensure_conversation(self, session_id: str, title: str = "Untitled") -> None:
        if not self._ready:
            return
        async with self._session() as db:
            async with db.begin():
                stmt = pg_insert(Conversation).values(
                    session_id=session_id,
                    title=title,
                    user_name="Aman",
                ).on_conflict_do_nothing(index_elements=["session_id"])
                await db.execute(stmt)

    async def load_messages(
        self, session_id: str, max_tokens: int = 4000
    ) -> list[BaseMessage]:
        """Load recent messages, respecting approximate token budget."""
        if not self._ready:
            return []

        async with self._session() as db:
            stmt = (
                select(PGMessage)
                .where(PGMessage.session_id == session_id)
                .order_by(PGMessage.id.desc())
                .limit(100)
            )
            rows = (await db.execute(stmt)).scalars().all()

        messages: list[BaseMessage] = []
        token_count = 0
        for row in rows:
            tokens = row.token_count or len(row.content) // 4
            if token_count + tokens > max_tokens:
                break
            token_count += tokens
            if row.role == "user":
                messages.append(HumanMessage(content=row.content))
            elif row.role == "assistant":
                messages.append(AIMessage(content=row.content))

        messages.reverse()  # chronological order
        return messages

    async def save_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        tool_calls: list[dict] | None = None,
    ) -> None:
        if not self._ready:
            return
        await self.ensure_conversation(session_id)

        async with self._session() as db:
            async with db.begin():
                db.add(PGMessage(
                    session_id=session_id,
                    role="user",
                    content=user_text,
                    token_count=len(user_text) // 4,
                ))
                db.add(PGMessage(
                    session_id=session_id,
                    role="assistant",
                    content=assistant_text,
                    token_count=len(assistant_text) // 4,
                    meta={"tool_calls": tool_calls} if tool_calls else {},
                ))

        logger.debug(f"💾 PG: saved turn for session '{session_id}'")

    async def clear_session(self, session_id: str) -> None:
        if not self._ready:
            return
        async with self._session() as db:
            async with db.begin():
                await db.execute(
                    delete(PGMessage).where(PGMessage.session_id == session_id)
                )

    # ── Facts ─────────────────────────────────────────────────────────────────

    async def save_fact(self, key: str, value: str, source: str = "conversation") -> None:
        if not self._ready:
            return
        async with self._session() as db:
            async with db.begin():
                stmt = pg_insert(Fact).values(
                    key=key, value=value, source=source
                ).on_conflict_do_update(
                    index_elements=["key"],
                    set_={"value": value, "source": source}
                )
                await db.execute(stmt)

    async def get_fact(self, key: str) -> str | None:
        if not self._ready:
            return None
        async with self._session() as db:
            row = (await db.execute(select(Fact).where(Fact.key == key))).scalar_one_or_none()
            return row.value if row else None

    async def get_all_facts(self) -> dict[str, str]:
        if not self._ready:
            return {}
        async with self._session() as db:
            rows = (await db.execute(select(Fact))).scalars().all()
            return {r.key: r.value for r in rows}

    # ── Personality state ─────────────────────────────────────────────────────

    async def get_personality(self) -> dict[str, float]:
        if not self._ready:
            return {"humor_level": 0.7, "verbosity": 0.5, "formality": 0.3}
        async with self._session() as db:
            row = (await db.execute(select(PersonalityState).where(PersonalityState.id == 1))).scalar_one_or_none()
            if not row:
                return {"humor_level": 0.7, "verbosity": 0.5, "formality": 0.3}
            return {
                "humor_level": row.humor_level,
                "verbosity": row.verbosity,
                "formality": row.formality,
            }

    async def update_personality(self, **kwargs: float) -> None:
        if not self._ready:
            return
        async with self._session() as db:
            async with db.begin():
                stmt = (
                    pg_insert(PersonalityState)
                    .values(id=1, **kwargs)
                    .on_conflict_do_update(index_elements=["id"], set_=kwargs)
                )
                await db.execute(stmt)
