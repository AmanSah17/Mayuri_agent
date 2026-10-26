"""
mayuri/memory/vector_store.py
─────────────────────────────
Vector store for semantic conversation compression and cross-thread recall.

Features:
  - Persistent ChromaDB vector storage on disk (data/chroma)
  - CUDA-accelerated sentence-transformers embeddings (all-MiniLM-L6-v2)
  - Compresses conversation turns and threads into dense semantic representations
  - Cross-thread retrieval: custom agent can recall context from any past session
  - Graceful fallback: if GPU is unavailable, automatically runs on CPU
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from mayuri.config import settings


class VectorConversationMemory:
    """
    Manages vector embeddings of conversation turns for semantic retrieval.
    """

    _instance: VectorConversationMemory | None = None

    def __init__(self) -> None:
        self._client: Any = None
        self._collection: Any = None
        self._ef: Any = None
        self._ready: bool = False
        self._init_lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> VectorConversationMemory:
        if cls._instance is None:
            cls._instance = VectorConversationMemory()
        return cls._instance

    def _init_sync(self) -> bool:
        if self._ready:
            return True
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            persist_dir = Path(settings.chroma_persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)

            device = settings.embedding_device
            try:
                import torch
                if device == "cuda" and not torch.cuda.is_available():
                    device = "cpu"
            except Exception:
                device = "cpu"

            self._ef = SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model_name,
                device=device,
            )

            self._client = chromadb.PersistentClient(path=str(persist_dir))
            self._collection = self._client.get_or_create_collection(
                name="conversation_threads",
                embedding_function=self._ef,
                metadata={"description": "Compressed conversation turns across sessions"},
            )
            self._ready = True
            logger.info(f"✅ Vector conversation memory active (device={device}, path={persist_dir})")
            return True
        except Exception as exc:
            logger.warning(f"⚠️ Vector memory initialization warning: {exc}")
            self._ready = False
            return False

    async def init(self) -> bool:
        """Initialize ChromaDB collection asynchronously."""
        async with self._init_lock:
            if self._ready:
                return True
            return await asyncio.to_thread(self._init_sync)

    def save_turn_sync(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        tool_names: list[str] | None = None,
    ) -> None:
        """Embed and store a compressed conversation turn."""
        if not self._ready and not self._init_sync():
            return

        try:
            now = time.time()
            doc_id = f"{session_id}_{int(now)}_{uuid.uuid4().hex[:6]}"

            # Compressed representation highlighting semantic content
            tools_str = f" [Used tools: {', '.join(tool_names)}]" if tool_names else ""
            compressed_doc = (
                f"User asked: {user_text.strip()}\n"
                f"Mayuri answered: {assistant_text.strip()}{tools_str}"
            )

            metadata = {
                "session_id": str(session_id),
                "timestamp": float(now),
                "user_snippet": user_text[:200],
                "assistant_snippet": assistant_text[:200],
            }

            self._collection.upsert(
                documents=[compressed_doc],
                metadatas=[metadata],
                ids=[doc_id],
            )
            logger.debug(f"🧠 Vector memory stored turn: {doc_id} (session={session_id})")
        except Exception as exc:
            logger.error(f"Failed to save turn to vector memory: {exc}")

    async def save_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        tool_names: list[str] | None = None,
    ) -> None:
        """Async wrapper to save turn to vector store."""
        await asyncio.to_thread(
            self.save_turn_sync, session_id, user_text, assistant_text, tool_names
        )

    def search_sync(
        self,
        query: str,
        n_results: int = 3,
        current_session: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for semantically relevant past conversation turns across all threads.
        """
        if not self._ready and not self._init_sync():
            return []

        try:
            count = self._collection.count()
            if count == 0:
                return []

            actual_k = min(n_results * 2, count)
            results = self._collection.query(
                query_texts=[query],
                n_results=actual_k,
            )

            hits: list[dict[str, Any]] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0] if "distances" in results else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                # Filter out exact duplicate of immediate query if needed
                hits.append({
                    "document": doc,
                    "session_id": meta.get("session_id", "unknown"),
                    "timestamp": meta.get("timestamp", 0.0),
                    "distance": dist,
                })

            # Return top-n
            return hits[:n_results]
        except Exception as exc:
            logger.warning(f"Vector search warning: {exc}")
            return []

    async def search(
        self,
        query: str,
        n_results: int = 3,
        current_session: str | None = None,
    ) -> list[dict[str, Any]]:
        """Async search wrapper."""
        return await asyncio.to_thread(self.search_sync, query, n_results, current_session)

    async def get_compressed_context(
        self,
        query: str,
        n_results: int = 3,
        current_session: str | None = None,
    ) -> str:
        """
        Retrieve compressed semantic memory and format it for prompt injection.
        """
        hits = await self.search(query, n_results=n_results, current_session=current_session)
        if not hits:
            return ""

        lines = ["## Recalled Past Conversations & Threads (Compressed Vector Memory)"]
        for i, hit in enumerate(hits, 1):
            sess = hit["session_id"]
            doc = hit["document"].replace("\n", " ")
            lines.append(f"{i}. [Session: {sess}] {doc}")

        return "\n".join(lines)
