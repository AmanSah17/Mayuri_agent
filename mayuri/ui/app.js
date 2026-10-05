/**
 * Mayuri UI — app.js
 * ───────────────────
 * Connects to the FastAPI WebSocket and drives the UI reactively.
 *
 * State machine:
 *   idle → listening → thinking → speaking → idle
 *
 * WebSocket events consumed:
 *   { type: "state",      data: { status: "listening"|"thinking"|"speaking"|"idle" } }
 *   { type: "transcript", data: { text: "..." } }
 *   { type: "response",   data: { text: "..." } }
 *   { type: "tool_call",  data: { tool: "...", args: {...} } }
 *   { type: "error",      data: { message: "..." } }
 */

'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
const API_BASE = 'http://127.0.0.1:8765';
const WS_URL   = 'ws://127.0.0.1:8765/ws/events';
const RECONNECT_DELAY_MS = 3000;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $body         = document.body;
const $orbLabel     = document.getElementById('orb-label');
const $statusDot    = document.getElementById('status-dot');
const $statusLabel  = document.getElementById('status-label');
const $chatFeed     = document.getElementById('chat-feed');
const $transcriptBar = document.getElementById('transcript-bar');
const $transcriptText = document.getElementById('transcript-text');
const $inputForm    = document.getElementById('input-form');
const $textInput    = document.getElementById('text-input');
const $footerModel  = document.getElementById('footer-model');
const $footerUptime = document.getElementById('footer-uptime');

// ── State ─────────────────────────────────────────────────────────────────────
let ws = null;
let reconnectTimer = null;
let currentState  = 'idle';
let uptimeStart   = Date.now();

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    clearTimeout(reconnectTimer);
    setConnectionStatus(true);
    // Keep-alive ping
    setInterval(() => ws && ws.readyState === WebSocket.OPEN && ws.send('ping'), 15_000);
  };

  ws.onclose = () => {
    setConnectionStatus(false);
    reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
  };

  ws.onerror = () => {
    setConnectionStatus(false);
  };

  ws.onmessage = (evt) => {
    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }
    handleEvent(msg.type, msg.data || {});
  };
}

// ── Event router ──────────────────────────────────────────────────────────────

function handleEvent(type, data) {
  switch (type) {
    case 'state':
      setState(data.status || 'idle');
      break;
    case 'transcript':
      showTranscript(data.text || '');
      appendMessage('user', data.text || '');
      break;
    case 'response':
      hideTranscript();
      appendMessage('ai', data.text || '');
      setState('idle');
      break;
    case 'tool_call':
      appendToolCall(data.tool, data.args || {});
      break;
    case 'error':
      appendMessage('ai', `⚠ ${data.message}`);
      setState('idle');
      break;
  }
}

// ── UI state machine ──────────────────────────────────────────────────────────

const STATE_LABELS = {
  idle:      'Idle',
  listening: 'Listening…',
  thinking:  'Thinking…',
  speaking:  'Speaking…',
};

function setState(state) {
  currentState = state;
  $body.dataset.state = state;
  $orbLabel.textContent = STATE_LABELS[state] || state;
}

// ── Connection status ─────────────────────────────────────────────────────────

function setConnectionStatus(connected) {
  $statusDot.className = 'status-dot' + (connected ? ' connected' : ' error');
  $statusLabel.textContent = connected ? 'Connected' : 'Reconnecting…';
}

// ── Transcript bar ────────────────────────────────────────────────────────────

function showTranscript(text) {
  $transcriptText.textContent = text;
  $transcriptBar.hidden = false;
  setState('thinking');
}

function hideTranscript() {
  $transcriptBar.hidden = true;
  $transcriptText.textContent = '';
}

// ── Chat feed ─────────────────────────────────────────────────────────────────

function appendMessage(role, text) {
  // Remove welcome message on first real message
  const welcome = $chatFeed.querySelector('.chat-welcome');
  if (welcome) welcome.remove();

  const wrap = document.createElement('div');
  wrap.className = `msg msg--${role === 'user' ? 'user' : 'ai'}`;

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = text;

  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  meta.textContent = role === 'user' ? 'You' : 'Mayuri';

  wrap.appendChild(bubble);
  wrap.appendChild(meta);
  $chatFeed.appendChild(wrap);
  $chatFeed.scrollTo({ top: $chatFeed.scrollHeight, behavior: 'smooth' });
}

function appendToolCall(toolName, args) {
  const badge = document.createElement('div');
  badge.className = 'tool-badge';
  badge.textContent = ` ${toolName}(${JSON.stringify(args).slice(0, 60)})`;
  $chatFeed.appendChild(badge);
  $chatFeed.scrollTo({ top: $chatFeed.scrollHeight, behavior: 'smooth' });
}

// ── Text input form ───────────────────────────────────────────────────────────

$inputForm.addEventListener('submit', async (evt) => {
  evt.preventDefault();
  const text = $textInput.value.trim();
  if (!text) return;
  $textInput.value = '';

  appendMessage('user', text);
  setState('thinking');

  try {
    const res = await fetch(`${API_BASE}/chat/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, session_id: 'default' }),
    });
    const data = await res.json();
    appendMessage('ai', data.response || 'No response.');
  } catch (err) {
    appendMessage('ai', `⚠ API error: ${err.message}`);
  } finally {
    setState('idle');
  }
});

// ── Footer uptime ─────────────────────────────────────────────────────────────

async function refreshStatus() {
  try {
    const res  = await fetch(`${API_BASE}/status`);
    const data = await res.json();
    const secs = Math.floor(data.uptime_seconds);
    const h = String(Math.floor(secs / 3600)).padStart(2, '0');
    const m = String(Math.floor((secs % 3600) / 60)).padStart(2, '0');
    const s = String(secs % 60).padStart(2, '0');
    $footerUptime.textContent = `Uptime: ${h}:${m}:${s}`;
  } catch { /* ignore */ }
}

async function loadConfig() {
  try {
    // Read model from Ollama tags endpoint
    const res  = await fetch('http://127.0.0.1:11434/api/tags');
    const data = await res.json();
    const models = (data.models || []).map(m => m.name).join(', ');
    $footerModel.textContent = `Model: ${models || '—'}`;
  } catch {
    $footerModel.textContent = 'Model: Ollama offline';
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────

connect();
loadConfig();
refreshStatus();
setInterval(refreshStatus, 30_000);
