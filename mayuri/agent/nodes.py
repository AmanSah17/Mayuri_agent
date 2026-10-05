"""
mayuri/agent/nodes.py
──────────────────────
LangGraph node functions.

Each node receives AgentState, does one unit of work, and returns a
partial state dict that LangGraph merges.

Nodes:
  llm_node      — calls Ollama, decides if a tool is needed
  tool_node     — dispatches MCP tool calls
  respond_node  — formats the final text answer
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from loguru import logger

from mayuri.agent.state import AgentState
from mayuri.config import settings


# ── LLM node ──────────────────────────────────────────────────────────────────

def llm_node(state: AgentState) -> dict:
    """
    Call Ollama with the current conversation.
    Returns an AIMessage (possibly with tool_calls attached).
    """
    from mayuri.agent.graph import build_llm  # lazy to avoid circular
    llm = build_llm()

    messages = state["messages"]
    response: AIMessage = llm.invoke(messages)

    logger.debug(f"LLM response: {response.content[:120]}")
    return {"messages": [response]}


# ── Tool router ───────────────────────────────────────────────────────────────

def should_use_tools(state: AgentState) -> str:
    """
    Conditional edge.
    Returns 'tools' if the last AIMessage has tool_calls, else 'respond'.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "respond"


# ── Tool execution node ────────────────────────────────────────────────────────

async def tool_node(state: AgentState) -> dict:
    """
    Execute MCP tool calls from the last AIMessage.
    Returns ToolMessages back into the conversation.
    """
    from mayuri.mcp.client import MCPClient

    last = state["messages"][-1]
    assert isinstance(last, AIMessage)

    tool_messages: list[ToolMessage] = []
    outputs: list[dict] = []

    client = MCPClient()

    for call in last.tool_calls:
        tool_name: str = call["name"]
        tool_args: dict = call["args"]
        tool_id: str = call["id"]

        logger.info(f"🔧 Tool call: {tool_name}({tool_args})")

        try:
            result = await client.call_tool(tool_name, tool_args)
        except Exception as exc:
            result = f"Error: {exc}"
            logger.error(f"Tool {tool_name} failed: {exc}")

        outputs.append({"tool": tool_name, "args": tool_args, "result": result})
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_id)
        )

    return {"messages": tool_messages, "tool_outputs": outputs}


# ── Final respond node ─────────────────────────────────────────────────────────

def respond_node(state: AgentState) -> dict:
    """
    Extract the final answer text from the conversation.
    The LLM already produced the answer — just surface it.
    """
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return {"final_answer": str(msg.content)}
    return {"final_answer": "I'm not sure how to help with that."}
