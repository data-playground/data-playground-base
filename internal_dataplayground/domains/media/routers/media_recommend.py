# routers/media_recommend.py
"""
Media Recommendations — ML pipeline with optional Gemini explanation layer.

Prefix: /media/recommend

Pipeline:
  1. Fetch user's liked items (rating >= 7) with their embeddings.
  2. Build composite query vector via ML service (/build_query).
  3. Fetch candidate items (want_to or untracked) with embeddings.
  4. Filter candidates by streaming service if include_unsubscribed=False.
  5. Find top-10 similar via ML service (/similarity).
  6. Optionally pass top-10 to Gemini for explanation and final top-5 selection.
     Controlled by Airflow Variable MEDIA_RECOMMEND_AI (true/false).
  7. Store results in media_recommendations table.
  8. Return partials/recommendations.html.

The Gemini toggle is read from the MEDIA_RECOMMEND_AI environment variable:
  - Set to "true" to enable Gemini explanation layer (adds ~2-3s latency).
  - Set to "false" to return pure ML results with similarity scores only.
  - If the variable is missing, defaults to "false" (safe default).
"""

import json
import logging
import os
from typing import Optional

from database import get_db
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from domains.media.models import (
    MediaItem,
    MediaRecommendation,
    RecommendationMediaType,
    StreamingService,
    UserMedia,
    UserMediaStatus,
)
from services.ai import MODEL_FLASH, call_gemini_json
from services.ml_service_client import build_query_vector, find_similar, health_check
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from routers._helpers import html_error
from core.templating import templates

router = APIRouter(prefix="/media/recommend", tags=["Media Recommendations"])
log = logging.getLogger(__name__)

# ── Gemini toggle ─────────────────────────────────────────────────────────────
# Read once at import time; change requires service restart.
# In Docker Compose, set: environment: - MEDIA_RECOMMEND_AI=true
_USE_GEMINI = os.environ.get("MEDIA_RECOMMEND_AI", "false").lower() == "true"
log.info("Recommendation Gemini layer: %s", "ENABLED" if _USE_GEMINI else "DISABLED")


@router.get("", response_class=HTMLResponse)
async def recommendations_page(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Renders the recommendation interface page."""
    # Check ML service health
    ml_available = await health_check()

    # Recent recommendation history
    history_result = await db.execute(
        select(MediaRecommendation)
        .order_by(MediaRecommendation.generated_at.desc())
        .limit(5)
    )
    history = history_result.scalars().all()

    # Subscribed services for the "only my services" toggle
    svc_result = await db.execute(
        select(StreamingService)
        .where(StreamingService.is_subscribed == True)
        .order_by(StreamingService.sort_order)
    )
    subscribed = svc_result.scalars().all()

    # Count liked items (to tell the user if they need to rate more)
    liked_result = await db.execute(
        select(UserMedia)
        .join(MediaItem)
        .where(UserMedia.user_rating >= 7)
        .where(MediaItem.embedding != None)
    )
    liked_count = len(liked_result.scalars().all())

    return templates.TemplateResponse("media_recommend.html", {
        "request": request,
        "active_module": "media",
        "ml_available": ml_available,
        "history": history,
        "subscribed_services": subscribed,
        "liked_count": liked_count,
        "gemini_enabled": _USE_GEMINI,
    })


@router.post("/generate", response_class=HTMLResponse)
async def generate_recommendations(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Runs the full recommendation pipeline and returns the recommendations partial.

    Form fields:
      media_type:          movie | tv_show | book | any
      mood:                optional free-text mood string
      occasion:            optional context string
      include_unsubscribed: "true" | "false" (default false)
    """
    form = await request.form()
    media_type_raw = form.get("media_type", "any")
    mood = (form.get("mood") or "").strip() or None
    occasion = (form.get("occasion") or "").strip() or None
    include_unsubscribed = form.get("include_unsubscribed", "false").lower() == "true"

    try:
        media_type_filter = RecommendationMediaType(media_type_raw)
    except ValueError:
        media_type_filter = RecommendationMediaType.ANY

    # ── Step 1: Fetch liked items with embeddings ─────────────────────────────
    liked_stmt = (
        select(UserMedia, MediaItem)
        .join(MediaItem, UserMedia.media_item_id == MediaItem.id)
        .where(UserMedia.user_rating >= 7)
        .where(MediaItem.embedding != None)
    )
    if media_type_filter != RecommendationMediaType.ANY:
        liked_stmt = liked_stmt.where(MediaItem.media_type == media_type_filter.value)

    liked_result = await db.execute(liked_stmt)
    liked_rows = liked_result.all()

    if not liked_rows:
        return html_error(
            request,
            "No rated items with embeddings found. "
            "Rate some items ≥ 7/10 and wait for the nightly embedding job to run.",
        )

    liked_items_payload = [
        {"embedding": row.MediaItem.embedding, "rating": row.UserMedia.user_rating}
        for row in liked_rows
    ]

    # ── Step 2: Build query vector ────────────────────────────────────────────
    try:
        query_vector = await build_query_vector(
            liked_items=liked_items_payload,
            mood_text=mood,
            mood_weight=0.3,
        )
    except Exception as exc:
        log.error("ML service build_query failed: %s", exc)
        return html_error(request, f"ML service unavailable: {exc}")

    # ── Step 3: Fetch candidates ──────────────────────────────────────────────
    # Candidates = items not yet completed/abandoned, with embeddings
    already_tracked_ids = {row.UserMedia.media_item_id for row in liked_rows}

    candidate_stmt = (
        select(MediaItem)
        .where(MediaItem.embedding != None)
        .where(MediaItem.id.not_in(already_tracked_ids))
    )
    if media_type_filter != RecommendationMediaType.ANY:
        candidate_stmt = candidate_stmt.where(
            MediaItem.media_type == media_type_filter.value
        )

    candidates_result = await db.execute(candidate_stmt)
    all_candidates = candidates_result.scalars().all()

    # ── Step 4: Filter by streaming service ───────────────────────────────────
    if not include_unsubscribed:
        # Get subscribed provider IDs
        svc_result = await db.execute(
            select(StreamingService.tmdb_provider_id)
            .where(StreamingService.is_subscribed == True)
            .where(StreamingService.tmdb_provider_id != None)
        )
        subscribed_ids = {row[0] for row in svc_result.all()}

        if subscribed_ids:
            filtered = []
            for c in all_candidates:
                available = set(c.streaming_available_on)
                # Books don't have streaming providers — always include them
                if c.media_type.value == "book" or available & subscribed_ids:
                    filtered.append(c)
            all_candidates = filtered

    if not all_candidates:
        return html_error(
            request,
            "No candidates with embeddings found. "
            "Try enabling 'Include non-subscribed services' or add more items.",
        )

    # ── Step 5: ML similarity ranking ─────────────────────────────────────────
    candidates_payload = [
        {"id": c.id, "embedding": c.embedding}
        for c in all_candidates
    ]

    try:
        similar = await find_similar(
            query_vector=query_vector,
            candidates=candidates_payload,
            top_n=10,
        )
    except Exception as exc:
        log.error("ML service similarity failed: %s", exc)
        return html_error(request, f"ML service error: {exc}")

    # Map IDs back to MediaItem objects
    candidate_map = {c.id: c for c in all_candidates}
    top_ml_items = [
        {"item": candidate_map[r["id"]], "score": r["score"]}
        for r in similar
        if r["id"] in candidate_map
    ]

    # ── Step 6: Optional Gemini explanation layer ─────────────────────────────
    recommendations = []
    used_gemini = False

    if _USE_GEMINI and top_ml_items:
        try:
            recommendations = await _gemini_explain(
                top_ml_items=top_ml_items,
                liked_rows=liked_rows,
                mood=mood,
                occasion=occasion,
            )
            used_gemini = True
        except Exception as exc:
            log.warning("Gemini explanation failed, falling back to ML-only: %s", exc)
            recommendations = _format_ml_only(top_ml_items[:5])
    else:
        recommendations = _format_ml_only(top_ml_items[:5])

    # ── Step 7: Cache results ─────────────────────────────────────────────────
    rec_record = MediaRecommendation(
        input_mood=mood,
        input_context=occasion,
        media_type_filter=media_type_filter,
        include_unsubscribed=include_unsubscribed,
        used_gemini=used_gemini,
        recommendations=recommendations,
        ml_model_version="all-MiniLM-L6-v2",
    )
    db.add(rec_record)
    await db.commit()
    await db.refresh(rec_record)

    # Load full MediaItem objects for the template
    rec_item_ids = [r["media_item_id"] for r in recommendations]
    items_result = await db.execute(
        select(MediaItem).where(MediaItem.id.in_(rec_item_ids))
    )
    items_map = {i.id: i for i in items_result.scalars().all()}

    # Check which are already tracked (want_to)
    tracked_result = await db.execute(
        select(UserMedia.media_item_id)
        .where(UserMedia.media_item_id.in_(rec_item_ids))
    )
    tracked_item_ids = {row[0] for row in tracked_result.all()}

    enriched = []
    for r in recommendations:
        mid = r["media_item_id"]
        item = items_map.get(mid)
        if item:
            enriched.append({
                **r,
                "media_item": item,
                "already_tracked": mid in tracked_item_ids,
            })

    return templates.TemplateResponse("partials/recommendations.html", {
        "request": request,
        "recommendations": enriched,
        "mood": mood,
        "occasion": occasion,
        "used_gemini": used_gemini,
        "rec_id": rec_record.id,
    })


@router.get("/history", response_class=HTMLResponse)
async def recommendation_history(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Shows past recommendation sessions."""
    result = await db.execute(
        select(MediaRecommendation)
        .order_by(MediaRecommendation.generated_at.desc())
        .limit(20)
    )
    history = result.scalars().all()

    return templates.TemplateResponse("media_rec_history.html", {
        "request": request,
        "active_module": "media",
        "history": history,
    })


# ── Internal helpers ──────────────────────────────────────────────────────────

def _format_ml_only(top_items: list[dict]) -> list[dict]:
    """Formats ML-only results into the standard recommendations JSON schema."""
    return [
        {
            "media_item_id": r["item"].id,
            "title": r["item"].title,
            "score": round(r["score"], 4),
            "reasoning": None,
        }
        for r in top_items
    ]


async def _gemini_explain(
    top_ml_items: list[dict],
    liked_rows: list,
    mood: Optional[str],
    occasion: Optional[str],
) -> list[dict]:
    """
    Asks Gemini to select the best 5 from the ML top-10 and explain each.

    Gemini sees:
      - The user's liked items (title + rating)
      - The ML top-10 candidates (title + similarity score)
      - The stated mood and occasion

    Gemini does NOT see embeddings or raw vectors — just titles and metadata.
    """
    # Build context
    liked_lines = "\n".join(
        f"- {row.MediaItem.title} ({row.MediaItem.release_year or '?'}) — rated {row.UserMedia.user_rating}/10"
        for row in liked_rows[:15]  # Cap context size
    )

    candidate_lines = "\n".join(
        f"{i+1}. {r['item'].title} ({r['item'].release_year or '?'}) "
        f"[{r['item'].media_type.value}] "
        f"[genres: {', '.join(r['item'].genre_list[:3])}] "
        f"[similarity: {r['score']:.3f}]"
        for i, r in enumerate(top_ml_items)
    )

    mood_context = ""
    if mood:
        mood_context += f"\nThe user wants: {mood}"
    if occasion:
        mood_context += f"\nContext: {occasion}"

    prompt = f"""You are a personal media recommendation assistant. A user's viewing/reading history has been analyzed by an ML similarity model.

Items the user has enjoyed (rated 7+/10):
{liked_lines}

Top 10 candidates from ML similarity analysis:
{candidate_lines}
{mood_context}

Select the best 5 from the 10 candidates for this user right now. For each, write ONE sentence of personalized reasoning that:
- References something specific from their history
- Explains why this fits the current mood/occasion (if stated)
- Is direct and honest — don't oversell

Respond ONLY with a JSON array, no markdown:
[
  {{"rank": 1, "candidate_number": N, "reasoning": "Because..."}},
  ...
]"""

    raw = call_gemini_json(prompt, schema=None, system=None, model=MODEL_FLASH)

    import json as _json
    import re
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    selections = _json.loads(cleaned)

    results = []
    for sel in selections[:5]:
        idx = sel.get("candidate_number", 1) - 1  # 1-indexed in prompt
        if 0 <= idx < len(top_ml_items):
            item = top_ml_items[idx]["item"]
            results.append({
                "media_item_id": item.id,
                "title": item.title,
                "score": round(top_ml_items[idx]["score"], 4),
                "reasoning": sel.get("reasoning"),
            })

    return results
