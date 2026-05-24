# ml_service/main.py
"""
Life OS — ML Service

A lightweight FastAPI container that wraps the sentence-transformers model.
Runs as a separate Docker container (ml-service) to isolate the heavy ML
dependency from the main FastAPI app and Airflow containers.

Called by:
  - Airflow DAG (life_os_generate_embeddings): batch embedding generation
  - FastAPI recommendation router: single-query embedding + similarity

API endpoints:
  POST /embed            — generate embedding for one or more texts
  POST /similarity       — compute top-N similar items from a query vector
  GET  /health           — liveness check for Docker healthcheck

Model: all-MiniLM-L6-v2
  - 384-dimensional output
  - ~80MB disk footprint
  - <100ms per item on CPU
  - Downloaded on first startup, cached to /app/model_cache/

The model is loaded once at startup (lifespan) and reused for all requests.
This avoids the ~2-3s cold start per request that loading on demand would cause.
"""

import logging
import math
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

log = logging.getLogger(__name__)

# ── Model state ───────────────────────────────────────────────────────────────
# Loaded once at startup, referenced globally within this process.
_model = None
_model_name = "all-MiniLM-L6-v2"
_model_version = f"sentence-transformers/{_model_name}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup, release on shutdown."""
    global _model
    log.info("Loading sentence-transformers model: %s", _model_name)
    from sentence_transformers import SentenceTransformer
    cache_dir = os.environ.get("MODEL_CACHE_DIR", "/app/model_cache")
    _model = SentenceTransformer(_model_name, cache_folder=cache_dir)
    log.info("Model loaded. Embedding dimension: %d", _model.get_sentence_embedding_dimension())
    yield
    _model = None
    log.info("Model released.")


app = FastAPI(
    title="Life OS ML Service",
    description="Sentence transformer embeddings for media recommendations",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ─────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    texts: list[str]
    """List of texts to embed. Each text gets one embedding vector."""


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    """One 384-dim float vector per input text, in the same order."""
    model_version: str
    dimension: int


class SimilarityRequest(BaseModel):
    query_vector: list[float]
    """Pre-computed query embedding vector (384-dim)."""
    candidates: list[dict]
    """List of {"id": any, "embedding": list[float]} dicts to rank."""
    top_n: int = 10
    """Number of top results to return."""


class SimilarityResult(BaseModel):
    id: int
    score: float


class SimilarityResponse(BaseModel):
    results: list[SimilarityResult]
    """Top-N candidates sorted by cosine similarity descending."""


class BuildQueryRequest(BaseModel):
    liked_items: list[dict]
    """[{"embedding": list[float], "rating": int}] — items rated >= 7."""
    mood_text: Optional[str] = None
    """Optional mood/craving string to bias the query vector."""
    mood_weight: float = 0.3
    """Weight of the mood vector relative to liked-item vectors. Default 0.3."""


class BuildQueryResponse(BaseModel):
    query_vector: list[float]
    """Weighted composite query vector, ready for similarity search."""


# ── Helper functions ──────────────────────────────────────────────────────────

def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Computes cosine similarity between two equal-length float vectors.
    Returns 0.0 if either vector is zero-magnitude.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _weighted_average(vectors: list[list[float]], weights: list[float]) -> list[float]:
    """
    Computes the weighted average of a list of float vectors.

    Args:
        vectors: List of equal-length float vectors.
        weights: Scalar weight for each vector (same length as vectors).

    Returns:
        Normalized weighted average vector.
    """
    if not vectors:
        return []

    dim = len(vectors[0])
    total_weight = sum(weights)
    if total_weight == 0:
        return [0.0] * dim

    result = [0.0] * dim
    for vec, w in zip(vectors, weights):
        for i, v in enumerate(vec):
            result[i] += v * w

    # Divide by total weight to get the weighted average
    return [r / total_weight for r in result]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Liveness check. Returns model status."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_version": _model_version,
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    Generates sentence embeddings for a list of text strings.

    The model runs synchronously on CPU. For batch embedding generation
    (Airflow DAG), pass multiple texts in one request to amortize model
    overhead. The model handles batching internally.

    Returns one 384-dim float vector per input text.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.texts:
        return EmbedResponse(embeddings=[], model_version=_model_version, dimension=384)

    # Run in the same thread — sentence-transformers is thread-safe
    # and the FastAPI event loop handles concurrency at the HTTP level.
    embeddings = _model.encode(
        request.texts,
        convert_to_numpy=False,
        show_progress_bar=False,
    )
    # Convert to plain Python float lists for JSON serialization
    return EmbedResponse(
        embeddings=[e.tolist() for e in embeddings],
        model_version=_model_version,
        dimension=len(embeddings[0]) if embeddings else 384,
    )


@app.post("/similarity", response_model=SimilarityResponse)
async def similarity(request: SimilarityRequest):
    """
    Computes cosine similarity between a query vector and a list of candidates.

    All similarity computation happens in Python (no model inference here).
    This endpoint exists to keep the ML container responsible for all vector
    math, keeping the main app free of numpy/scipy dependencies.

    Args:
        query_vector: 384-dim composite query vector (from /build_query or
                      pre-computed by the caller).
        candidates:   List of {"id": int, "embedding": list[float]} dicts.
        top_n:        Number of top results to return.

    Returns:
        top_n candidates sorted by cosine similarity descending.
    """
    if not request.candidates:
        return SimilarityResponse(results=[])

    scored = []
    for candidate in request.candidates:
        emb = candidate.get("embedding")
        if not emb or len(emb) == 0:
            continue
        score = _cosine_similarity(request.query_vector, emb)
        scored.append(SimilarityResult(id=candidate["id"], score=round(score, 6)))

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x.score, reverse=True)
    return SimilarityResponse(results=scored[: request.top_n])


@app.post("/build_query", response_model=BuildQueryResponse)
async def build_query(request: BuildQueryRequest):
    """
    Builds the composite query vector for a recommendation request.

    Algorithm:
      1. For each liked item (rating >= 7 enforced by caller), compute weight
         as (rating - 6). Rating 10 → weight 4, rating 7 → weight 1.
      2. Compute weighted average of liked-item embedding vectors.
      3. If mood_text is provided, embed it and add it to the composite
         with mood_weight relative to the liked-items total weight.
      4. Return the composite vector for use in /similarity.

    Args:
        liked_items: [{"embedding": [...], "rating": int}]
        mood_text:   Optional mood/craving string ("something light and funny")
        mood_weight: Relative weight of the mood vector (default 0.3)

    Returns:
        Composite query vector ready for /similarity.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.liked_items:
        raise HTTPException(status_code=400, detail="No liked items provided")

    # Filter items that have embeddings
    valid = [
        item for item in request.liked_items
        if item.get("embedding") and len(item["embedding"]) > 0
    ]
    if not valid:
        raise HTTPException(status_code=400, detail="No items with embeddings found")

    # Step 1 & 2: weighted average of liked items
    vectors = [item["embedding"] for item in valid]
    weights = [max(0.1, item.get("rating", 7) - 6) for item in valid]
    composite = _weighted_average(vectors, weights)

    # Step 3: optionally blend in mood embedding
    if request.mood_text and request.mood_text.strip():
        mood_embedding = _model.encode(
            [request.mood_text.strip()],
            convert_to_numpy=False,
            show_progress_bar=False,
        )[0].tolist()

        # Blend: composite * (1 - mood_weight) + mood * mood_weight
        w = max(0.0, min(0.9, request.mood_weight))  # clamp to [0, 0.9]
        composite = [
            c * (1 - w) + m * w
            for c, m in zip(composite, mood_embedding)
        ]

    return BuildQueryResponse(query_vector=composite)
