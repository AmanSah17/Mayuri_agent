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
import re
import uuid
from functools import lru_cache
from textwrap import dedent

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from mayuri.agent.state import AgentState, PersonalityConfig
from mayuri.agent.tools import ALL_TOOLS
from mayuri.api.ws.events import EventBus
from mayuri.config import settings
from mayuri.memory.pg_store import PGMemoryStore
from mayuri.memory.store import MemoryStore
from mayuri.memory.vector_store import VectorConversationMemory


# ── Personality prompt builder ────────────────────────────────────────────────

def _build_system_prompt(
    personality: PersonalityConfig,
    facts: dict[str, str],
    recalled_context: str = "",
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
        - play_music: play songs, videos, or music on YouTube or Spotify.
        - open_app: launch desktop apps (Chrome, Notepad, Calculator, Spotify, VS Code, etc.).
        - open_browser: open websites and URLs in the default browser.
        - tavily_search: real-time web search (news, facts, prices)
        - exa_search: neural/semantic web search (research, deep content)
        - quick_answer: fast factual Q&A
        - get_datetime: current date and time in IST
        - calculate: safe math evaluation

        ## Core rules
        1. When Aman asks to play a song, track, video, or music (e.g. "play Kalyani", "play music by Shreya Ghoshal", "play thunderstruck", "play on youtube"), YOU MUST CALL the `play_music` tool immediately. NEVER write markdown URLs like `[Song](https://youtube.com/...)` or fake URLs like "example_video_id" in conversational text!
        2. When Aman asks to open an app (e.g. "open notepad", "open chrome"), call `open_app`.
        3. When Aman asks to open a website, call `open_browser`.
        4. Never execute arbitrary code — only use named tools.
        5. If you don't know something, use a search tool rather than guessing.
        6. Keep spoken answers concise — you can elaborate if asked.
        7. Be honest. Never fabricate facts or URLs.
        8. If a tool fails or reports an issue, DO NOT get stuck in a retry loop or repeatedly apologize. Immediately bypass the failing tool, switch to an alternative tool or use your general knowledge, and deliver a clean answer.
        {facts_str}
        {recalled_context}
    """).strip()


# ── LLM factory ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_llm_with_tools() -> ChatOllama:
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0.8,
        streaming=True,
    )
    return llm.bind_tools(ALL_TOOLS)


def build_llm() -> ChatOllama:
    """Public helper returning the initialized Ollama LLM bound with tools."""
    return _get_llm_with_tools()


# ── Fallback intent recovery ──────────────────────────────────────────────────

def _recover_missing_tool_call(user_input: str, response_content: str) -> list[dict]:
    """
    If the LLM failed to emit a structured tool call and instead hallucinated
    an action (e.g. 'I will play Thunderstruck on YouTube', or user explicitly asked
    to play music / open an app), recover by synthesizing the proper tool call.
    """
    text_lower = user_input.lower().strip()
    resp_lower = response_content.lower() if response_content else ""

    # Check 1: Play music / YouTube request
    play_match = re.search(
        r'\b(?:play\s+(?:on\s+)?(?:youtube\s+)?(?:song\s+)?|listen\s+to\s+)(.+?)(?:,?\s*(?:open\s+youtube|on\s+youtube|on\s+spotify|on\s+browser)|$)',
        text_lower
    )
    if "example_video_id" in resp_lower or ("play" in text_lower and ("youtube" in text_lower or "spotify" in text_lower or "song" in text_lower or "music" in text_lower)):
        query = ""
        if play_match and play_match.group(1).strip():
            query = play_match.group(1).strip()
            query = re.sub(r'\b(on\s+youtube|on\s+browser|on\s+spotify|please|song|track)\b', '', query, flags=re.IGNORECASE).strip()

        if not query and response_content:
            quoted = re.findall(r'["\']([^"\']+)["\']', response_content)
            if quoted:
                query = quoted[0]

        if not query:
            query = re.sub(r'\b(play|on|youtube|browser|spotify|music|song)\b', '', text_lower).strip()

        if query:
            logger.info(f"🛠️ Recovered missing play_music tool call for: '{query}'")
            return [{
                "name": "play_music",
                "args": {"query": query, "platform": "youtube"},
                "id": f"call_play_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }]

    # Check 2: Open app request (e.g. "open notepad", "open chrome")
    app_match = re.search(r'\b(?:open|launch|start)\s+(notepad|calculator|calc|chrome|spotify|vscode|code)\b', text_lower)
    if app_match:
        app_name = app_match.group(1).strip()
        logger.info(f"🛠️ Recovered missing open_app tool call for: '{app_name}'")
        return [{
            "name": "open_app",
            "args": {"name": app_name},
            "id": f"call_app_{uuid.uuid4().hex[:8]}",
            "type": "tool_call",
        }]

    return []


def _clean_for_speech(text: str) -> str:
    """Strip markdown links, URLs, and code characters so TTS speaks natural English."""
    # Convert markdown links [Title](url) -> Title
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove raw URLs
    cleaned = re.sub(r'https?://\S+', '', cleaned)
    # Remove markdown asterisks, backticks, hash symbols
    cleaned = re.sub(r'[*_`#]', '', cleaned)
    # Clean redundant spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


# ── Graph nodes ───────────────────────────────────────────────────────────────

def llm_node(state: AgentState) -> dict:
    """Call Ollama with personality-aware system prompt and full history."""
    llm = _get_llm_with_tools()

    # Rebuild system prompt every turn (personality may have changed)
    sys_prompt = _build_system_prompt(
        state.get("personality", {"humor_level": 0.7, "verbosity": 0.5, "formality": 0.3}),
        state.get("user_facts", {}),
        state.get("recalled_context", ""),
    )

    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    response: AIMessage = llm.invoke(messages)

    # Intent recovery: ONLY check if no tool has run yet this turn (prevents infinite loop)
    has_tool_run = any(isinstance(m, ToolMessage) for m in state.get("messages", []))
    if not has_tool_run and not response.tool_calls:
        recovered = _recover_missing_tool_call(state.get("user_input", ""), response.content or "")
        if recovered:
            response.tool_calls = recovered

    model_name = response.response_metadata.get("model", settings.ollama_model)
    logger.info(f"🤖 LLM [{model_name}]: {response.content[:120] if response.content else '[tool call]'}")

    # Broadcast real-time tool calls to connected frontend assistant UI
    if response.tool_calls:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                for call in response.tool_calls:
                    asyncio.create_task(
                        EventBus.emit("tool_call", {"tool": call.get("name"), "args": call.get("args")})
                    )
        except Exception:
            pass

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
    """Extract final answer text from the last AIMessage, cleaned for speech."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            raw_text = str(msg.content)
            clean_speech = _clean_for_speech(raw_text)
            return {"final_answer": clean_speech, "needs_tools": False}
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
      - Vector store (ChromaDB + SentenceTransformers): compressed semantic memory across all threads
      - PostgreSQL: long-term structured history + facts + personality (persists across restarts)
      - SQLite: fast session cache (in-memory speed for recent context)
    """

    _pg: PGMemoryStore | None = None
    _sqlite: MemoryStore | None = None
    _vector: VectorConversationMemory | None = None

    @classmethod
    async def get_stores(cls) -> tuple[PGMemoryStore, MemoryStore, VectorConversationMemory]:
        if cls._pg is None:
            cls._pg = PGMemoryStore()
            await cls._pg.init()
        if cls._sqlite is None:
            cls._sqlite = MemoryStore()
        if cls._vector is None:
            cls._vector = VectorConversationMemory.get_instance()
            await cls._vector.init()
        return cls._pg, cls._sqlite, cls._vector

    async def run(self, user_input: str, session_id: str = "default") -> str:
        """
        Run one conversational turn.

        Returns the assistant's response text (to be spoken by TTS).
        """
        pg, sqlite, vector = await self.get_stores()

        # Load personality and facts from PostgreSQL
        personality = await pg.get_personality()
        user_facts = await pg.get_all_facts()

        # Semantic vector recall: retrieve compressed memory from previous conversations and threads
        recalled_context = ""
        try:
            recalled_context = await vector.get_compressed_context(
                query=user_input,
                n_results=3,
                current_session=session_id,
            )
        except Exception as exc:
            logger.warning(f"Vector memory recall error: {exc}")

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
            "recalled_context": recalled_context,
            "tool_outputs": [],
            "final_answer": "",
            "needs_tools": False,
        }

        logger.info(f"🧠 Agent: '{user_input[:80]}' (session={session_id})")
        result: AgentState = await _graph.ainvoke(initial)

        answer = result.get("final_answer", "")

        # Persist to all memory stores
        tool_outputs = result.get("tool_outputs", [])
        tool_names = [t.get("tool", "") for t in tool_outputs if isinstance(t, dict)]
        await pg.save_turn(session_id, user_input, answer, tool_outputs or None)
        await sqlite.save_turn(session_id, user_input, answer)
        try:
            await vector.save_turn(session_id, user_input, answer, tool_names=tool_names or None)
        except Exception as exc:
            logger.warning(f"Vector store persistence error: {exc}")

        return answer
