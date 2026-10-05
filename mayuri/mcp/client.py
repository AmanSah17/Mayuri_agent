"""
mayuri/mcp/client.py
─────────────────────
MCP client layer.

Connects to all running MCP servers and presents their tools as
LangChain-compatible tools that LangGraph can call.

MCP servers are separate processes (stdio transport). The client
starts them as subprocesses and communicates over stdin/stdout.

Server registry (mcp_servers.json) defines which servers to launch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

# MCP server registry — loaded from config file
_SERVERS_CONFIG = Path(__file__).parent / "mcp_servers.json"


class MCPClient:
    """Manages connections to MCP servers and tool dispatching."""

    def __init__(self) -> None:
        self._servers = self._load_servers_config()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_langchain_tools(self) -> list:
        """
        Return all tools from all MCP servers as LangChain BaseTool objects.
        Used by build_llm() to bind tools to the Ollama model.
        """
        try:
            from langchain_mcp_adapters.tools import load_mcp_tools  # type: ignore[import]
        except ImportError:
            logger.warning(
                "langchain-mcp-adapters not installed — tools disabled. "
                "pip install langchain-mcp-adapters"
            )
            return []

        tools = []
        for name, cfg in self._servers.items():
            try:
                server_tools = load_mcp_tools(cfg)
                tools.extend(server_tools)
                logger.info(f"Loaded {len(server_tools)} tools from MCP server '{name}'")
            except Exception as exc:
                logger.error(f"Failed to load tools from '{name}': {exc}")

        return tools

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """
        Dispatch a single tool call to the appropriate MCP server.

        Routing: each tool name is prefixed with server name (e.g. "browser__navigate").
        """
        # Find which server handles this tool
        server_prefix = tool_name.split("__")[0] if "__" in tool_name else None

        for name, cfg in self._servers.items():
            if server_prefix is None or name == server_prefix:
                try:
                    return await self._call_on_server(cfg, tool_name, args)
                except Exception as exc:
                    logger.error(f"Tool call {tool_name} on server {name} failed: {exc}")
                    raise

        raise ValueError(f"No MCP server found for tool: {tool_name}")

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_servers_config() -> dict:
        if not _SERVERS_CONFIG.exists():
            logger.warning(f"MCP servers config not found: {_SERVERS_CONFIG}")
            return {}
        with _SERVERS_CONFIG.open() as f:
            return json.load(f)

    @staticmethod
    async def _call_on_server(cfg: dict, tool_name: str, args: dict) -> Any:
        """Low-level MCP stdio call."""
        from mcp import ClientSession, StdioServerParameters  # type: ignore[import]
        from mcp.client.stdio import stdio_client  # type: ignore[import]

        params = StdioServerParameters(command=cfg["command"], args=cfg.get("args", []))

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, args)
                return result
