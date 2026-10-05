"""
mayuri/mcp/servers/browser_server.py
──────────────────────────────────────
MCP server: Browser automation via Playwright.

Exposes tools:
  browser__navigate(url)              — Navigate to a URL
  browser__get_text()                 — Get visible page text
  browser__click(selector)            — Click an element
  browser__type(selector, text)       — Type into an element
  browser__screenshot()               — Take a screenshot (base64 PNG)
  browser__search_web(query)          — DuckDuckGo search, return top results
  browser__summarize_page(url)        — Navigate + extract main content

Run as: python -m mayuri.mcp.servers.browser_server
"""

from __future__ import annotations

import asyncio
import base64
import json

from mcp.server import Server  # type: ignore[import]
from mcp.server.stdio import stdio_server  # type: ignore[import]
from mcp.types import TextContent, Tool  # type: ignore[import]
from loguru import logger

app = Server("browser")
_browser = None
_page = None


async def _get_page():
    global _browser, _page
    if _page is None:
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        _browser = await pw.chromium.launch(headless=False)
        context = await _browser.new_context()
        _page = await context.new_page()
    return _page


# ── Tool definitions ──────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="browser__navigate",
            description="Navigate the browser to a URL",
            inputSchema={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL to navigate to"}},
                "required": ["url"],
            },
        ),
        Tool(
            name="browser__get_text",
            description="Get all visible text content from the current page",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="browser__click",
            description="Click an element on the page",
            inputSchema={
                "type": "object",
                "properties": {"selector": {"type": "string"}},
                "required": ["selector"],
            },
        ),
        Tool(
            name="browser__type",
            description="Type text into an input element",
            inputSchema={
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["selector", "text"],
            },
        ),
        Tool(
            name="browser__screenshot",
            description="Take a screenshot of the current browser page",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="browser__search_web",
            description="Search DuckDuckGo and return top results as text",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        Tool(
            name="browser__summarize_page",
            description="Navigate to a URL and return the main text content",
            inputSchema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        ),
    ]


# ── Tool handlers ─────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    page = await _get_page()

    if name == "browser__navigate":
        await page.goto(arguments["url"], wait_until="domcontentloaded")
        return [TextContent(type="text", text=f"Navigated to {arguments['url']}")]

    elif name == "browser__get_text":
        text = await page.inner_text("body")
        return [TextContent(type="text", text=text[:8000])]

    elif name == "browser__click":
        await page.click(arguments["selector"])
        return [TextContent(type="text", text=f"Clicked {arguments['selector']}")]

    elif name == "browser__type":
        await page.fill(arguments["selector"], arguments["text"])
        return [TextContent(type="text", text=f"Typed into {arguments['selector']}")]

    elif name == "browser__screenshot":
        png = await page.screenshot()
        encoded = base64.b64encode(png).decode()
        return [TextContent(type="text", text=f"data:image/png;base64,{encoded}")]

    elif name == "browser__search_web":
        query = arguments["query"]
        url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}&ia=web"
        await page.goto(url, wait_until="domcontentloaded")
        results = await page.query_selector_all(".result__body")
        texts = []
        for r in results[:5]:
            t = await r.inner_text()
            texts.append(t.strip())
        return [TextContent(type="text", text="\n\n".join(texts))]

    elif name == "browser__summarize_page":
        await page.goto(arguments["url"], wait_until="domcontentloaded")
        # Try to extract main content
        for selector in ["article", "main", "#content", ".content", "body"]:
            el = await page.query_selector(selector)
            if el:
                text = await el.inner_text()
                return [TextContent(type="text", text=text[:6000])]

    return [TextContent(type="text", text="Unknown tool")]


# ── Entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
