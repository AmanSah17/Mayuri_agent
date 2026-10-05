"""
mayuri/agent/graph.py
──────────────────────
Mayuri LangGraph agent with:
  - ReAct tool-calling loop (Think → Act → Observe → Answer)
  - Personality-aware system prompt (humor_level, verbosity, formality)
  - Long-term memory from PostgreSQL (facts + history)
  - LangSmith tracing (auto-enabled via env vars)
  - Tavily + Exa web search tools
  - Conversation thread continuity across sessions
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from textwrap import dedent

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from mayuri.agent.state import AgentState, PersonalityConfig
from mayuri.agent.tools import ALL_TOOLS
from mayuri.config import settings
from mayuri.memory.pg_store import PGMemoryStore
from mayuri.memory.store import MemoryStore


# ── Personality prompt builder ────────────────────────────────────────────────

def _build_system_prompt(
    personality: PersonalityConfig,
    facts: dict[str, str],
) -> str:
    """
    Dynamically build the system prompt based on personality state and known facts.
    humor_level 0→1 shifts tone from formal/neutral to witty/playful.
    """
    h = personality.get("humor_level", 0.7)
    v = personality.get("verbosity", 0.5)
    f_level = personality.get("formality", 0.3)

    # Tone instructions scaled by humor_level
    if h >= 0.8:
        tone = dedent("""
            You have a playful, witty personality. You enjoy light banter, clever observations,
            and occasional self-deprecating humor. You can tease Aman affectionately (he likes it).
            Use casual contractions. Occasionally add a joke or a fun fact when appropriate.
            You sound like a brilliant friend who happens to know everything.
        """)
    elif h >= 0.5:
        tone = dedent("""
            You are warm and friendly with a dry sense of humor. You're helpful and precise,
            but you don't take yourself too seriously. You occasionally drop a witty remark.
            You call Aman by name and speak naturally.
        """)
    else:
        tone = dedent("""
            You are professional and precise. Minimal small talk. Get to the point efficiently.
            You are polite but not overly friendly.
        """)

    # Verbosity instructions
    if v >= 0.7:
        verbosity_note = "Give thorough, detailed answers. Explain your reasoning."
    elif v >= 0.4:
        verbosity_note = "Be concise but complete. One to three sentences for simple questions."
    else:
        verbosity_note = "Be extremely concise. One sentence maximum unless the task requires more."

    # Facts section
    facts_str = ""
    if facts:
        facts_lines = "\n".join(f"  • {k}: {v}" for k, v in facts.items())
        facts_str = f"\n\n## Things you know about Aman:\n{facts_lines}"

    return dedent(f"""
        You are Mayuri, a local-first AI assistant running on Aman's Windows machine.
        You are always-available, privacy-first, and extraordinarily capable.

        ## Personality
        {tone.strip()}

        ## Response style
        {verbosity_note}
        - You are speaking aloud via TTS — avoid markdown, bullet points, or code blocks
          in spoken responses unless the user is reading text.
        - Address the user as "Aman" naturally (not every sentence).
        - Never say "As an AI..." — just answer.

        ## Capabilities
        You have access to:
        - tavily_search: real-time web search (news, facts, prices)
        - exa_search: neural/semantic web search (research, deep content)
        - quick_answer: fast factual Q&A
        - get_datetime: current date and time in IST
        - calculate: safe math evaluation

        ## Core rules
        1. Never execute arbitrary code — only use named tools.
        2. If you don't know something, use a search tool rather than guessing.
        3. Keep spoken answers concise — you can elaborate if asked.
        4. Be honest. Never fabricate facts or URLs.
        {facts_str}
    """).strip()


# ── LLM factory ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_llm_with_tools() -> ChatOllama:
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.6,
        streaming=True,
    )
    return llm.bind_tools(ALL_TOOLS)


# ── Graph nodes ───────────────────────────────────────────────────────────────

def llm_node(state: AgentState) -> dict:
    """Call Ollama with personality-aware system prompt and full history."""
    llm = _get_llm_with_tools()

    # Rebuild system prompt every turn (personality may have changed)
    sys_prompt = _build_system_prompt(
        state.get("personality", {"humor_level": 0.7, "verbosity": 0.5, "formality": 0.3}),
        state.get("user_facts", {}),
    )

    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    response: AIMessage = llm.invoke(messages)

    logger.debug(f"LLM: {response.content[:120] if response.content else '[tool call]'}")
    return {
        "messages": [response],
        "needs_tools": bool(response.tool_calls),
    }


def should_use_tools(state: AgentState) -> str:
    """Conditional edge: route to tools or respond."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "respond"


def respond_node(state: AgentState) -> dict:
    """Extract final answer text from the last AIMessage."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return {"final_answer": str(msg.content), "needs_tools": False}
    return {"final_answer": "I'm not sure how to help with that.", "needs_tools": False}


# ── Graph construction ────────────────────────────────────────────────────────

def _build_graph() -> object:
    tool_node = ToolNode(ALL_TOOLS)

    builder = StateGraph(AgentState)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "llm")
    builder.add_conditional_edges(
        "llm",
        should_use_tools,
        {"tools": "tools", "respond": "respond"},
    )
    builder.add_edge("tools", "llm")   # loop: observe → think → act/answer
    builder.add_edge("respond", END)

    return builder.compile()


_graph = _build_graph()


# ── MayuriAgent: high-level interface ─────────────────────────────────────────

class MayuriAgent:
    """
    Unified agent interface used by audio pipeline and REST API.

    Memory strategy:
      - PostgreSQL: long-term history + facts + personality (persists across restarts)
      - SQLite: fast session cache (in-memory speed for recent context)
    """

    _pg: PGMemoryStore | None = None
    _sqlite: MemoryStore | None = None

    @classmethod
    async def get_stores(cls) -> tuple[PGMemoryStore, MemoryStore]:
        if cls._pg is None:
            cls._pg = PGMemoryStore()
            await cls._pg.init()
        if cls._sqlite is None:
            cls._sqlite = MemoryStore()
        return cls._pg, cls._sqlite

    async def run(self, user_input: str, session_id: str = "default") -> str:
        """
        Run one conversational turn.

        Returns the assistant's response text (to be spoken by TTS).
        """
        pg, sqlite = await self.get_stores()

        # Load personality and facts from PostgreSQL
        personality = await pg.get_personality()
        user_facts = await pg.get_all_facts()

        # Load recent conversation history (PostgreSQL → richer long-term context)
        pg_history = await pg.load_messages(session_id, max_tokens=3000)
        # Also load from SQLite for speed (recent turns)
        sqlite_history = await sqlite.load_messages(session_id)

        # Merge: prefer PG history, top up with SQLite if PG is empty
        history = pg_history if pg_history else sqlite_history

        initial: AgentState = {
            "messages": [*history, HumanMessage(content=user_input)],
            "user_input": user_input,
            "session_id": session_id,
            "personality": personality,
            "user_facts": user_facts,
            "tool_outputs": [],
            "final_answer": "",
            "needs_tools": False,
        }

        logger.info(f"🧠 Agent: '{user_input[:80]}' (session={session_id})")
        result: AgentState = await _graph.ainvoke(initial)

        answer = result.get("final_answer", "")

        # Persist to both stores
        tool_outputs = result.get("tool_outputs", [])
        await pg.save_turn(session_id, user_input, answer, tool_outputs or None)
        await sqlite.save_turn(session_id, user_input, answer)

        return answer
