"""
mayuri/memory/pg_models.py
───────────────────────────
SQLAlchemy ORM models for PostgreSQL long-term memory.

Tables:
  conversations        — Named conversation sessions with metadata
  messages             — Full message history per conversation
  facts                — Long-term user knowledge (name, prefs, etc.)
  personality_state    — Mayuri's current humor/mood calibration
  embeddings           — Vector embeddings for semantic recall (pgvector)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Float, ForeignKey,
    Integer, String, Text, UniqueConstraint, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB


class PGBase(DeclarativeBase):
    pass


class Conversation(PGBase):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(512), default="Untitled")
    user_name: Mapped[str] = mapped_column(String(128), default="Aman")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    meta: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    messages: Mapped[list["PGMessage"]] = relationship(
        "PGMessage", back_populates="conversation",
        cascade="all, delete-orphan", order_by="PGMessage.id"
    )


class PGMessage(PGBase):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("conversations.session_id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user | assistant | tool | system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tool_name: Mapped[Optional[str]] = mapped_column(String(128))
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    meta: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    conversation: Mapped[Conversation] = relationship("Conversation", back_populates="messages")


class Fact(PGBase):
    __tablename__ = "facts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(256), unique=True, index=True, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(128), default="conversation")  # conversation|manual
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PersonalityState(PGBase):
    """Mayuri's current humor/tone calibration — updated by interactions."""
    __tablename__ = "personality_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    humor_level: Mapped[float] = mapped_column(Float, default=0.7)     # 0=serious, 1=playful
    verbosity: Mapped[float] = mapped_column(Float, default=0.5)        # 0=terse, 1=verbose
    formality: Mapped[float] = mapped_column(Float, default=0.3)        # 0=casual, 1=formal
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
