"""
mayuri/agent/tools.py
──────────────────────
All LangChain-compatible tools available to Mayuri.

Tools are grouped:
  1. Web search  — Tavily (general) + Exa (neural/semantic)
  2. Browser     — via MCP browser_server
  3. System      — via MCP windows_server + filesystem_server

The LLM sees these as structured function calls — never raw exec().
"""

from __future__ import annotations

from langchain_core.tools import tool
from loguru import logger

from mayuri.config import settings


# ── 1. Tavily web search ──────────────────────────────────────────────────────

@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily for real-time, accurate information.
    Use this for: news, facts, prices, current events, product info.

    Args:
        query: The search query string.
        max_results: Number of results to return (default 5).
    """
    try:
        from tavily import TavilyClient  # type: ignore[import]
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=True,
        )
        lines = []
        if response.get("answer"):
            lines.append(f"Direct answer: {response['answer']}\n")
        for r in response.get("results", []):
            lines.append(f"• {r['title']}\n  {r['url']}\n  {r.get('content', '')[:300]}\n")
        return "\n".join(lines) or "No results found."
    except Exception as exc:
        logger.error(f"Tavily search failed: {exc}")
        return f"Search failed: {exc}"


# ── 2. Exa neural search ──────────────────────────────────────────────────────

@tool
def exa_search(query: str, num_results: int = 5, use_autoprompt: bool = True) -> str:
    """
    Search the web using Exa's neural search for semantically relevant results.
    Best for: research papers, in-depth articles, finding specific URLs.

    Args:
        query: Natural language search query.
        num_results: Number of results (default 5).
        use_autoprompt: Let Exa rephrase the query for better results.
    """
    try:
        from exa_py import Exa  # type: ignore[import]
        client = Exa(api_key=settings.exa_api_key)
        result = client.search_and_contents(
            query,
            num_results=num_results,
            use_autoprompt=use_autoprompt,
            text={"max_characters": 800},
        )
        lines = []
        for r in result.results:
            lines.append(f"• {r.title}\n  {r.url}\n  {(r.text or '')[:400]}\n")
        return "\n".join(lines) or "No results found."
    except Exception as exc:
        logger.error(f"Exa search failed: {exc}")
        return f"Exa search failed: {exc}"


# ── 3. Quick web answer (Tavily with answer mode) ─────────────────────────────

@tool
def quick_answer(question: str) -> str:
    """
    Get a quick direct answer to a factual question using web search.
    Best for: "What is X?", "Who invented Y?", "When did Z happen?"

    Args:
        question: A factual question to answer.
    """
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.qna_search(query=question)
        return str(response)
    except Exception as exc:
        logger.error(f"Quick answer failed: {exc}")
        return f"Couldn't get an answer: {exc}"


# ── 4. Get current date/time ──────────────────────────────────────────────────

@tool
def get_datetime() -> str:
    """
    Get the current date and time in India (IST timezone).
    Use this whenever the user asks about the current time or date.
    """
    from datetime import datetime
    import zoneinfo
    ist = zoneinfo.ZoneInfo("Asia/Kolkata")
    now = datetime.now(ist)
    return now.strftime("Today is %A, %d %B %Y. Current time: %I:%M %p IST.")


# ── 5. Calculator ─────────────────────────────────────────────────────────────

@tool
def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Use for: arithmetic, unit conversions, percentages.
    Never use for code execution.

    Args:
        expression: Math expression like "2 ** 10", "1500 * 0.18", "sqrt(144)"
    """
    import math
    allowed_names = {
        k: v for k, v in math.__dict__.items() if not k.startswith("__")
    }
    allowed_names["abs"] = abs
    allowed_names["round"] = round

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Calculation error: {exc}"


# ── Tool registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = [
    tavily_search,
    exa_search,
    quick_answer,
    get_datetime,
    calculate,
]
