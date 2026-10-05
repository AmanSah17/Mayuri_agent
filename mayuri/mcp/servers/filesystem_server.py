"""
mayuri/mcp/servers/filesystem_server.py
─────────────────────────────────────────
MCP server: File system operations.

Exposes tools:
  filesystem__read_file(path)                 — Read file contents
  filesystem__write_file(path, content)       — Write/overwrite a file
  filesystem__list_dir(path)                  — List directory contents
  filesystem__search_files(query, directory)  — Find files by name pattern
  filesystem__delete_file(path)               — Delete a file (with confirm)
  filesystem__open_file(path)                 — Open file with default app

Safety: All paths are validated to be within allowed directories.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import subprocess
from pathlib import Path

from mcp.server import Server  # type: ignore[import]
from mcp.server.stdio import stdio_server  # type: ignore[import]
from mcp.types import TextContent, Tool  # type: ignore[import]
from loguru import logger

app = Server("filesystem")

# Only allow operations within these directories for safety
ALLOWED_ROOTS = [
    Path.home(),
    Path("D:/"),
    Path("C:/Users"),
]


def _validate_path(path_str: str) -> Path:
    p = Path(path_str).resolve()
    for root in ALLOWED_ROOTS:
        try:
            p.relative_to(root.resolve())
            return p
        except ValueError:
            continue
    raise PermissionError(f"Access denied: {path_str} is outside allowed directories.")


# ── Tool definitions ──────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="filesystem__read_file",
            description="Read the contents of a file",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        Tool(
            name="filesystem__write_file",
            description="Write content to a file (creates if not exists)",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        ),
        Tool(
            name="filesystem__list_dir",
            description="List files and directories at a path",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        Tool(
            name="filesystem__search_files",
            description="Search for files matching a name pattern",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Filename pattern e.g. *.py"},
                    "directory": {"type": "string"},
                },
                "required": ["query", "directory"],
            },
        ),
        Tool(
            name="filesystem__open_file",
            description="Open a file with its default Windows application",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
    ]


# ── Tool handlers ─────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "filesystem__read_file":
            p = _validate_path(arguments["path"])
            content = p.read_text(encoding="utf-8", errors="replace")
            return [TextContent(type="text", text=content[:10_000])]

        elif name == "filesystem__write_file":
            p = _validate_path(arguments["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(arguments["content"], encoding="utf-8")
            return [TextContent(type="text", text=f"Written: {p}")]

        elif name == "filesystem__list_dir":
            p = _validate_path(arguments["path"])
            entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
            lines = [f"{'[DIR] ' if e.is_dir() else '      '}{e.name}" for e in entries]
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "filesystem__search_files":
            root = _validate_path(arguments["directory"])
            pattern = arguments["query"]
            matches = [
                str(p) for p in root.rglob("*")
                if fnmatch.fnmatch(p.name, pattern)
            ][:50]
            return [TextContent(type="text", text="\n".join(matches) or "No matches found.")]

        elif name == "filesystem__open_file":
            p = _validate_path(arguments["path"])
            os.startfile(str(p))  # Windows only
            return [TextContent(type="text", text=f"Opened: {p}")]

    except PermissionError as exc:
        return [TextContent(type="text", text=f"Permission denied: {exc}")]
    except Exception as exc:
        return [TextContent(type="text", text=f"Error: {exc}")]

    return [TextContent(type="text", text="Unknown tool")]


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
