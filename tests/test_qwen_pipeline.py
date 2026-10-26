"""
tests/test_qwen_pipeline.py
───────────────────────────
Integration and verification test suite confirming the Mayuri pipeline
is utilizing the local Qwen model (qwen2.5:3b) via Ollama.

Tests:
  1. Ollama service health & Qwen model presence in local Ollama tags.
  2. LangChain ChatOllama direct invocation & model metadata verification.
  3. Qwen ReAct tool calling verification.
  4. LangGraph MayuriAgent full turn execution with Qwen.
  5. AudioPipeline integration (process_text -> Qwen agent -> event dispatch).
  6. Status route reporting active Qwen model health.

Run with:
  & D:\\CUDA_ENV\\CUDA_ENV\\Scripts\\Activate.ps1; python -m pytest tests/test_qwen_pipeline.py -v -s
"""

from __future__ import annotations

import asyncio
import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mayuri.config import settings


@pytest.mark.asyncio
async def test_ollama_qwen_available():
    """Verify Ollama is reachable and the configured Qwen model is present."""
    import httpx

    url = f"{settings.ollama_base_url}/api/tags"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=5.0)

    assert resp.status_code == 200, f"Ollama returned {resp.status_code} at {url}"
    data = resp.json()
    model_names = [m.get("name", "") for m in data.get("models", [])]
    print(f"\n[OK] Available models in Ollama: {model_names}")

    is_found = any(settings.ollama_model in name for name in model_names)
    assert is_found, (
        f"Configured model '{settings.ollama_model}' not found in Ollama tags: {model_names}. "
        f"Run: ollama pull {settings.ollama_model}"
    )
    print(f"   Target model '{settings.ollama_model}' is present.")


@pytest.mark.asyncio
async def test_qwen_llm_direct_invocation():
    """Verify ChatOllama invokes Qwen and reports the exact model metadata."""
    from mayuri.agent.graph import build_llm

    llm = build_llm()
    prompt = "State your model family and say hello in under 10 words."
    response = await asyncio.to_thread(llm.invoke, prompt)

    model_used = response.response_metadata.get("model", "")
    print(f"\n[OK] LLM response: '{response.content.strip()}'")
    print(f"   Model reported by Ollama: '{model_used}'")

    assert response.content, "LLM returned empty response"
    assert "qwen" in model_used.lower(), (
        f"Expected model name to contain 'qwen', got '{model_used}'"
    )


@pytest.mark.asyncio
async def test_qwen_tool_calling():
    """Verify that Qwen is able to bind tools and produce structured tool calls."""
    from langchain_core.messages import HumanMessage
    from mayuri.agent.graph import build_llm

    llm = build_llm()
    message = HumanMessage(content="Calculate 25 * 4 using the calculate tool.")
    response = await asyncio.to_thread(llm.invoke, [message])

    print(f"\n[OK] Tool calling response: content='{response.content}', tool_calls={response.tool_calls}")
    # Qwen should either produce a tool call or answer 100 directly
    has_tool_call = bool(response.tool_calls)
    has_answer = "100" in response.content

    assert has_tool_call or has_answer, (
        f"Qwen neither emitted a tool call nor computed the answer. Tool calls: {response.tool_calls}, Content: {response.content}"
    )


@pytest.mark.asyncio
async def test_mayuri_agent_utilizing_qwen():
    """Verify MayuriAgent LangGraph orchestrator runs with Qwen."""
    from mayuri.agent.graph import MayuriAgent

    agent = MayuriAgent()
    answer = await agent.run(
        user_input="Introduce yourself briefly in one sentence.",
        session_id="test_qwen_pipeline",
    )

    print(f"\n[OK] Agent response: '{answer}'")
    assert answer, "MayuriAgent returned empty answer"
    assert len(answer) > 5


@pytest.mark.asyncio
async def test_audio_pipeline_integration_with_qwen():
    """Verify the AudioPipeline routes text through the Qwen agent end-to-end."""
    from mayuri.audio.pipeline import AudioPipeline

    pipeline = AudioPipeline()
    response = await pipeline.process_text(
        text="What is 50 plus 50?",
        speak=False,
        session_id="test_pipeline_e2e",
    )

    print(f"\n[OK] AudioPipeline response: '{response}'")
    assert response, "AudioPipeline returned empty response"
    assert "100" in response, f"Expected 100 in response, got '{response}'"


@pytest.mark.asyncio
async def test_system_status_reports_qwen_ok():
    """Verify the /status endpoint recognizes that the Qwen model is active."""
    from mayuri.api.routes.status import get_status

    status = await get_status()
    ollama_comp = next((c for c in status.components if c.name == "ollama"), None)

    assert ollama_comp is not None, "Ollama component missing from status response"
    print(f"\n[OK] Status endpoint Ollama component: status={ollama_comp.status}, detail='{ollama_comp.detail}'")
    assert ollama_comp.status == "ok"
    assert "loaded" in ollama_comp.detail
