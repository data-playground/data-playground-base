# routers/_recommend_pipeline.py
"""
Media Recommendations — scoring/explanation internals.

Split out of media_recommend.py (WO#28 Part A) to keep that router under
the 300-line limit. Pure relocation — no behavior change.

Holds the two helpers that turn ML top-10 similarity results into the
final recommendations JSON schema:
  - _format_ml_only(): ML-only path, no LLM call.
  - _gemini_explain(): asks Gemini to pick + explain the final top-5.

Both are called from generate_recommendations() in media_recommend.py,
which still owns the _USE_GEMINI toggle and the asyncio.to_thread(...)
wrapping decision (see that module — this file only holds the internals
that were wrapped, not the wrapping decision itself).
"""

import asyncio
from typing import Optional

from services.ai import MODEL_FLASH, call_gemini_json


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

    # Runs off the event loop: call_gemini_json is a synchronous call
    # (blocking requests.post, plus blocking time.sleep() on any 503
    # retry). _gemini_explain() is itself async and awaited from
    # generate_recommendations(), but without this, a single slow or
    # retried Gemini call would stall every other request this FastAPI
    # worker is handling, not just this one.
    raw = await asyncio.to_thread(call_gemini_json, prompt, schema=None, system=None, model=MODEL_FLASH)

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
