"""
mayuri/agent/state.py
──────────────────────
Extended LangGraph state schema for the Mayuri agent.

Includes:
  - Full conversation thread
  - Personality calibration
  - Tool outputs scratch pad
  - Humor/mood parameters carried per-turn
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class PersonalityConfig(TypedDict):
    humor_level: float    # 0.0 = serious, 1.0 = maximum banter
    verbosity: float      # 0.0 = one-liners, 1.0 = detailed
    formality: float      # 0.0 = "yo", 1.0 = "certainly"


class AgentState(TypedDict):
    """Immutable-ish state carried through the LangGraph nodes."""

    # Full conversation history — LangGraph merges via add_messages
    messages: Annotated[list[BaseMessage], add_messages]

    # Raw user input (post-STT, post-wake-word-strip)
    user_input: str

    # Session identifier for memory routing
    session_id: str

    # Personality parameters loaded from PostgreSQL
    personality: PersonalityConfig

    # All known facts about the user (from PG facts table)
    user_facts: dict[str, str]

    # Tool call outputs accumulated this turn
    tool_outputs: list[dict]

    # Set by respond_node — the final spoken text
    final_answer: str

    # Internal flag: did the LLM request tools this step?
    needs_tools: bool
