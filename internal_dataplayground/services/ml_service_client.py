# services/ml_service_client.py
"""
HTTP client for the Life OS ML service container.

The ML service runs at http://ml-service:8001 inside the Docker network.
All embedding generation and similarity computation routes through here
so neither the main FastAPI app nor Airflow DAGs carry the sentence-transformers
dependency directly.

FastAPI routers use the async functions (await).
Airflow DAG tasks use the sync wrapper functions (no event loop needed).
"""

import json
import logging
import os
from typing import Optional

import httpx

log = logging.getLogger(__name__)

ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://ml-service:8001")


# ── ASYNC (used by FastAPI routers) ───────────────────────────────────────────

async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generates embeddings for a list of texts via the ML service.
    Returns a list of 384-dim float vectors in the same order as input.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{ML_SERVICE_URL}/embed",
            json={"texts": texts},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]


async def build_query_vector(
    liked_items: list[dict],
    mood_text: Optional[str] = None,
    mood_weight: float = 0.3,
) -> list[float]:
    """
    Builds a weighted composite query vector from liked items.

    Args:
        liked_items: [{"embedding": list[float], "rating": int}]
        mood_text:   Optional mood string to bias the query.
        mood_weight: Weight of mood relative to liked-item vectors.

    Returns:
        384-dim composite query vector.
    """
    payload = {
        "liked_items": liked_items,
        "mood_text": mood_text,
        "mood_weight": mood_weight,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{ML_SERVICE_URL}/build_query",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["query_vector"]


async def find_similar(
    query_vector: list[float],
    candidates: list[dict],
    top_n: int = 10,
) -> list[dict]:
    """
    Finds the top-N most similar candidates to a query vector.

    Args:
        query_vector: 384-dim float vector (from build_query_vector).
        candidates:   [{"id": int, "embedding": list[float]}]
        top_n:        Number of results to return.

    Returns:
        [{"id": int, "score": float}] sorted by score descending.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{ML_SERVICE_URL}/similarity",
            json={
                "query_vector": query_vector,
                "candidates": candidates,
                "top_n": top_n,
            },
        )
        resp.raise_for_status()
        return resp.json()["results"]


async def health_check() -> bool:
    """Returns True if the ML service is up and model is loaded."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ML_SERVICE_URL}/health")
            return resp.status_code == 200 and resp.json().get("model_loaded", False)
    except Exception:
        return False


# ── SYNC (used by Airflow DAG tasks — no asyncio event loop) ─────────────────

def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    """Synchronous version of embed_texts for use in Airflow task callables."""
    resp = httpx.post(
        f"{ML_SERVICE_URL}/embed",
        json={"texts": texts},
        timeout=120.0,  # Longer timeout for large batches
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def health_check_sync() -> bool:
    """Synchronous health check for Airflow pre-flight."""
    try:
        resp = httpx.get(f"{ML_SERVICE_URL}/health", timeout=5.0)
        return resp.status_code == 200 and resp.json().get("model_loaded", False)
    except Exception:
        return False


def build_embedding_text(item: dict) -> str:
    """
    Constructs the text string to embed for a media_item.
    This is the canonical embedding input — used by both the DAG and the router.
    Keeping it here (not in the DAG) ensures both callers produce identical text.

    Args:
        item: dict with keys: title, genres (list), description, author (optional)

    Returns:
        Pipe-separated string of metadata fields.
    """
    parts = [
        item.get("title", ""),
    ]
    genres = item.get("genres") or []
    if genres:
        parts.append(f"Genre: {', '.join(genres)}")
    description = (item.get("description") or "")[:500]
    if description:
        parts.append(description)
    if item.get("author"):
        parts.append(f"Author: {item['author']}")
    if item.get("media_type"):
        parts.append(f"Type: {item['media_type']}")
    return " | ".join(p for p in parts if p)
