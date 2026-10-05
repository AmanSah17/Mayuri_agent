# Mayuri — Local-first AI Operating Layer for Windows

```
"Mayuri, search Amazon for RTX 5070 laptops."
```

Mayuri is an always-listening, privacy-first AI assistant that runs **entirely on your local machine**. It combines a tiny wake-word model (openWakeWord) that is always on with a sleeping LLM (Ollama) that only wakes when you say **"Mayuri"**.

---

## Architecture

```
ALWAYS RUNNING (tiny footprint)
┌────────────────────────────────┐
│  Microphone → openWakeWord     │
└───────────────┬────────────────┘
                │ "Mayuri" detected
                ▼
ACTIVATED PIPELINE
┌────────────────────────────────────────────────────────┐
│  VAD (Silero) → faster-whisper STT                     │
│       ↓                                                 │
│  FastAPI Backend  ←→  WebSocket ←→  Desktop UI         │
│       ↓                                                 │
│  LangGraph Agent Orchestrator                           │
│       ↓                                                 │
│  MCP Client → [browser | filesystem | web | windows]   │
│       ↓                                                 │
│  Ollama (local LLM)                                     │
│       ↓                                                 │
│  Piper TTS → Speaker                                    │
└────────────────────────────────────────────────────────┘
```

## Project Layout

```
mayuri/
├── mayuri/                     # Main Python package
│   ├── main.py                 # Entry point — launches all services
│   ├── config.py               # Pydantic settings
│   ├── wake/                   # Wake-word detection (always on)
│   │   ├── service.py
│   │   └── detector.py
│   ├── audio/                  # Microphone capture + VAD
│   │   ├── capture.py
│   │   └── vad.py
│   ├── stt/                    # Speech-to-text (faster-whisper)
│   │   └── transcriber.py
│   ├── tts/                    # Text-to-speech (Piper)
│   │   └── synthesizer.py
│   ├── agent/                  # LangGraph orchestrator + Ollama
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── state.py
│   ├── mcp/                    # MCP client + server stubs
│   │   ├── client.py
│   │   └── servers/
│   │       ├── browser_server.py
│   │       ├── filesystem_server.py
│   │       ├── web_search_server.py
│   │       └── windows_server.py
│   ├── memory/                 # SQLite conversation memory
│   │   ├── models.py
│   │   └── store.py
│   ├── api/                    # FastAPI backend
│   │   ├── server.py
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   ├── status.py
│   │   │   └── memory.py
│   │   └── ws/
│   │       └── events.py
│   └── ui/                     # Desktop UI (JavaScript/HTML)
│       ├── app.js
│       ├── index.html
│       └── style.css
├── scripts/                    # Helper scripts
│   ├── install_models.py
│   └── setup_windows_service.py
├── tests/
├── pyproject.toml
├── .env.example
└── README.md
```

## Quick Start

```bash
# 1. Clone and enter
git clone <repo> && cd mayuri

# 2. Create virtual environment
python -m venv .venv && .venv\Scripts\activate

# 3. Install
pip install -e ".[dev]"

# 4. Install Playwright browsers
playwright install chromium

# 5. Download models
python scripts/install_models.py

# 6. Configure
copy .env.example .env   # edit as needed

# 7. Run
mayuri
```

## Key Design Principles

| Principle | Detail |
|-----------|--------|
| **Privacy first** | No data leaves the machine |
| **Tiny always-on** | Only wake-word model runs continuously |
| **No exec()** | LLM never runs arbitrary code — structured tool calls only |
| **MCP tools** | Browser, filesystem, Windows APIs, web search |
| **Streaming** | WebSocket pushes every token to UI in real time |
| **Modular** | Each service is an independent process |
