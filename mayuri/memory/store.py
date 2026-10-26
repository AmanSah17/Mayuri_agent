"""
mayuri/memory/store.py
───────────────────────
Async SQLite-backed conversation memory.

Usage:
    store = MemoryStore()
    history = await store.load_messages("default")
    await store.save_turn("default", "what's the weather?", "It's 25°C in Bangalore.")
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from mayuri.config import settings
from mayuri.memory.models import Base, Message, Session


class MemoryStore:
    """Async SQLite conversation memory using SQLAlchemy + aiosqlite."""

    def __init__(self) -> None:
        db_path = settings.memory_db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )
        self._initialized = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def _ensure_tables(self) -> None:
        if not self._initialized:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._initialized = True

    # ── Public API ────────────────────────────────────────────────────────────

    async def load_messages(self, session_id: str) -> list[BaseMessage]:
        """Load the last N turns from a session as LangChain messages."""
        await self._ensure_tables()

        async with self._session_factory() as db:
            stmt = (
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.id.desc())
                .limit(settings.memory_max_history * 2)
            )
            rows = (await db.execute(stmt)).scalars().all()

        # Reverse to chronological order
        messages: list[BaseMessage] = []
        for row in reversed(rows):
            if row.role == "user":
                messages.append(HumanMessage(content=row.content))
            elif row.role == "assistant":
                messages.append(AIMessage(content=row.content))

        return messages

    async def save_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """Persist one user→assistant turn."""
        await self._ensure_tables()

        async with self._session_factory() as db:
            async with db.begin():
                # Ensure session exists
                stmt_sess = select(Session).where(Session.session_id == session_id)
                existing = (await db.execute(stmt_sess)).scalar_one_or_none()
                if existing is None:
                    db.add(Session(session_id=session_id))

                db.add(Message(session_id=session_id, role="user", content=user_text))
                db.add(Message(session_id=session_id, role="assistant", content=assistant_text))

        logger.debug(f"Memory saved for session '{session_id}'")

    async def save_fact(self, key: str, value: str) -> None:
        """Store or update a long-term fact about the user."""
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        from mayuri.memory.models import Fact

        await self._ensure_tables()
        async with self._session_factory() as db:
            async with db.begin():
                stmt = sqlite_insert(Fact).values(key=key, value=value)
                stmt = stmt.on_conflict_do_update(index_elements=["key"], set_={"value": value})
                await db.execute(stmt)

    async def get_fact(self, key: str) -> str | None:
        """Retrieve a long-term fact by key."""
        from mayuri.memory.models import Fact

        await self._ensure_tables()
        async with self._session_factory() as db:
            stmt = select(Fact).where(Fact.key == key)
            row = (await db.execute(stmt)).scalar_one_or_none()
            return row.value if row else None

    async def clear_session(self, session_id: str) -> None:
        """Delete all messages in a session."""
        from sqlalchemy import delete
        from mayuri.memory.models import Message

        await self._ensure_tables()
        async with self._session_factory() as db:
            async with db.begin():
                await db.execute(delete(Message).where(Message.session_id == session_id))
