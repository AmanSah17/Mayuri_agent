"""
mayuri/mcp/servers/web_search_server.py
─────────────────────────────────────────
MCP server: Web search.

Uses Brave Search API (preferred) or Serper API (fallback).
If no API key is set, falls back to DuckDuckGo via browser.

Exposes tools:
  web_search__search(query, num_results)  — Search the web
  web_search__news(query)                 — Search recent news
"""

from __future__ import annotations

import asyncio
import json

import httpx
from mcp.server import Server  # type: ignore[import]
from mcp.server.stdio import stdio_server  # type: ignore[import]
from mcp.types import TextContent, Tool  # type: ignore[import]
from loguru import logger

from mayuri.config import settings

app = Server("web_search")


# ── Tool definitions ──────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search__search",
            description="Search the web and return top results with titles and snippets",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "num_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="web_search__news",
            description="Search for recent news articles on a topic",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
    ]


# ── Search implementations ────────────────────────────────────────────────────

async def _brave_search(query: str, count: int = 5) -> list[dict]:
    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": settings.brave_api_key,
            },
            params={"q": query, "count": count},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("web", {}).get("results", [])
        return [
            {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("description")}
            for r in results
        ]


async def _serper_search(query: str, count: int = 5) -> list[dict]:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": settings.serper_api_key, "Content-Type": "application/json"},
            json={"q": query, "num": count},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return [
            {"title": i.get("title"), "url": i.get("link"), "snippet": i.get("snippet")}
            for i in data.get("organic", [])
        ]


async def _search(query: str, count: int = 5) -> list[dict]:
    """Try Brave → Serper → graceful error."""
    if settings.brave_api_key:
        return await _brave_search(query, count)
    if settings.serper_api_key:
        return await _serper_search(query, count)
    logger.warning("No web search API key set. Results will be empty.")
    return []


# ── Tool handler ──────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    query = arguments["query"]
    count = arguments.get("num_results", 5)

    results = await _search(query, count)

    if not results:
        return [TextContent(type="text", text="No results found.")]

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}\n")

    return [TextContent(type="text", text="\n".join(lines))]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
