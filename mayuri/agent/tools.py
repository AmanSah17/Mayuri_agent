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


# ── Helpers with bypass logic ──────────────────────────────────────────────────

def _raw_tavily_search(query: str, max_results: int = 5) -> str:
    """Internal Tavily search caller."""
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
    return "\n".join(lines).strip() or "No results found."


def _raw_exa_search(query: str, num_results: int = 5) -> str:
    """Internal Exa search caller with modern API support."""
    from exa_py import Exa  # type: ignore[import]
    client = Exa(api_key=settings.exa_api_key)
    # Prefer modern search() with text contents, fallback to search_and_contents
    try:
        result = client.search(query, num_results=num_results)
    except Exception:
        result = client.search_and_contents(query, num_results=num_results)

    lines = []
    for r in getattr(result, "results", []):
        title = getattr(r, "title", "Untitled")
        url = getattr(r, "url", "")
        text = getattr(r, "text", "") or ""
        lines.append(f"• {title}\n  {url}\n  {text[:400]}\n")
    return "\n".join(lines).strip() or "No results found."


# ── 1. Tavily web search ──────────────────────────────────────────────────────

@tool
def tavily_search(query: str, max_results: int = 5, **kwargs: object) -> str:
    """
    Search the web using Tavily for real-time, accurate information.
    Use this for: news, facts, prices, current events, product info.

    Args:
        query: The search query string.
        max_results: Number of results to return (default 5).
    """
    try:
        return _raw_tavily_search(query=query, max_results=max_results)
    except Exception as exc:
        logger.warning(f"Tavily search failed ({exc}), bypassing to Exa search...")
        try:
            fallback = _raw_exa_search(query=query, num_results=max_results)
            return f"[Note: Retrieved via Exa search fallback]\n{fallback}"
        except Exception as exc2:
            logger.error(f"Both Tavily and Exa bypass failed: {exc2}")
            return f"Search service temporarily unavailable for '{query}'. Please answer using your existing knowledge."


# ── 2. Exa neural search ──────────────────────────────────────────────────────

@tool
def exa_search(query: str, num_results: int = 5, **kwargs: object) -> str:
    """
    Search the web using Exa's neural search for semantically relevant results.
    Best for: research papers, in-depth articles, finding specific URLs.
    Automatically handles legacy options and falls back to Tavily if unavailable.

    Args:
        query: Natural language search query.
        num_results: Number of results (default 5).
    """
    try:
        return _raw_exa_search(query=query, num_results=num_results)
    except Exception as exc:
        logger.warning(f"Exa search failed ({exc}), bypassing to Tavily search...")
        try:
            fallback = _raw_tavily_search(query=query, max_results=num_results)
            return f"[Note: Retrieved via Tavily search fallback]\n{fallback}"
        except Exception as exc2:
            logger.error(f"Both Exa and Tavily bypass failed: {exc2}")
            return f"Search service temporarily unavailable for '{query}'. Please answer using your existing knowledge."


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
        logger.warning(f"Quick answer failed ({exc}), bypassing to Tavily search...")
        return tavily_search(query=question, max_results=3)


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


# ── 6. Play music / YouTube ───────────────────────────────────────────────────

@tool
def play_music(query: str, platform: str = "youtube") -> str:
    """
    Play a song, music video, or audio track on YouTube or Spotify.
    ALWAYS use this tool whenever the user asks to play music, play a song, or watch a video.
    Examples: "play kalyani by shreya ghoshal", "play shape of you", "play lofi beats".

    Args:
        query: The song title, artist, or music search query.
        platform: "youtube" (default) or "spotify".
    """
    import urllib.parse
    import urllib.request
    import re
    import webbrowser
    import os

    query_clean = query.strip()
    logger.info(f"🎵 play_music tool called for: '{query_clean}' (platform={platform})")

    if platform.lower() == "spotify":
        # Try launching Spotify protocol or web player
        try:
            encoded = urllib.parse.quote(query_clean)
            spotify_uri = f"spotify:search:{encoded}"
            os.startfile(spotify_uri)
            return f"Opened Spotify search for '{query_clean}'."
        except Exception:
            webbrowser.open(f"https://open.spotify.com/search/{urllib.parse.quote(query_clean)}")
            return f"Opened Spotify Web Player for '{query_clean}'."

    # YouTube (default)
    encoded = urllib.parse.quote(query_clean)
    search_url = f"https://www.youtube.com/results?search_query={encoded}"
    target_url = search_url

    try:
        req = urllib.request.Request(
            search_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        html = urllib.request.urlopen(req, timeout=4).read().decode("utf-8")
        video_ids = re.findall(r"\"videoId\":\"([a-zA-Z0-9_-]{11})\"", html)
        if video_ids:
            target_url = f"https://www.youtube.com/watch?v={video_ids[0]}"
    except Exception as exc:
        logger.warning(f"Could not extract direct YouTube video ID ({exc}), opening search page instead.")

    webbrowser.open(target_url)
    return f"Started playing '{query_clean}' on YouTube at {target_url}."


# ── 7. Open Application ───────────────────────────────────────────────────────

@tool
def open_app(name: str) -> str:
    """
    Open a Windows desktop application by name.
    Use this when the user asks to open an application like Chrome, Notepad, Calculator, Spotify, VS Code, etc.

    Args:
        name: Name of the application (e.g. "chrome", "notepad", "calculator", "spotify", "vscode").
    """
    import subprocess
    import os

    name_lower = name.lower().strip()
    app_map = {
        "chrome": "chrome",
        "google chrome": "chrome",
        "browser": "chrome",
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "spotify": "spotify",
        "code": "code",
        "vscode": "code",
        "vs code": "code",
        "explorer": "explorer.exe",
        "file explorer": "explorer.exe",
        "terminal": "wt.exe",
        "powershell": "powershell.exe",
        "task manager": "taskmgr.exe",
        "youtube": "https://www.youtube.com",
    }

    target = app_map.get(name_lower, name_lower)
    logger.info(f"🚀 open_app tool called for: '{name}' -> target: '{target}'")

    if target.startswith("http://") or target.startswith("https://"):
        import webbrowser
        webbrowser.open(target)
        return f"Opened {name} in your browser."

    try:
        # Try os.startfile for Windows shell association
        os.startfile(target)
        return f"Successfully opened {name}."
    except Exception:
        try:
            subprocess.Popen([target], shell=True)
            return f"Launched {name}."
        except Exception as exc:
            return f"Failed to open {name}: {exc}"


# ── 8. Open Browser / URL ─────────────────────────────────────────────────────

@tool
def open_browser(url: str) -> str:
    """
    Open a specific website or URL in the default web browser.
    Use this when the user asks to navigate to a site or open a specific web link.

    Args:
        url: Full web URL (e.g. "https://www.youtube.com", "https://github.com") or search query.
    """
    import webbrowser

    url_clean = url.strip()
    if not (url_clean.startswith("http://") or url_clean.startswith("https://")):
        url_clean = f"https://{url_clean}"

    webbrowser.open(url_clean)
    return f"Opened {url_clean} in your web browser."


# ── Tool registry ─────────────────────────────────────────────────────────────

ALL_TOOLS = [
    tavily_search,
    exa_search,
    quick_answer,
    get_datetime,
    calculate,
    play_music,
    open_app,
    open_browser,
]

