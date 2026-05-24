# routers/media_search.py
"""
Media Search — TMDB (movies/TV) and OpenLibrary (books) search with add-to-list.

Prefix: /media/search

The search flow:
  1. User types in the search bar → debounced HTMX GET to /media/search/query
  2. Results appear as cards in partials/search_results.html
  3. "Add" button → POST /media/search/add → creates media_item + user_media
  4. Streaming providers are fetched and stored when creating the media_item

All external API calls are async and use services/tmdb_service.py
and services/openlibrary_service.py.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import MediaItem, UserMedia, MediaType, UserMediaStatus, MediaExternalSource
from routers._helpers import html_error
from services import tmdb_service, openlibrary_service

router = APIRouter(prefix="/media/search", tags=["Media Search"])
templates = Jinja2Templates(directory="templates")
log = logging.getLogger(__name__)


@router.get("", response_class=HTMLResponse)
async def search_page(request: Request):
    """Renders the search interface page."""
    return templates.TemplateResponse("media_search.html", {
        "request": request,
        "active_module": "media",
    })


@router.get("/query", response_class=HTMLResponse)
async def search_query(
    request: Request,
    db: AsyncSession = Depends(get_db),
    q: str = "",
    type: str = "movie",  # movie | tv | book
):
    """
    Searches TMDB or OpenLibrary and returns result cards.
    Called by HTMX on debounced input. Returns partials/search_results.html.

    For each result, we check if the item is already in the user's list
    so the "Add" button can show "Already tracking" instead.
    """
    if not q or len(q.strip()) < 2:
        return HTMLResponse('<p style="color:var(--text-muted);font-size:11px;padding:8px 0;">Type at least 2 characters to search.</p>')

    q = q.strip()
    results = []

    try:
        if type == "movie":
            results = await tmdb_service.search_movies(q)
        elif type == "tv":
            results = await tmdb_service.search_tv(q)
        elif type == "book":
            results = await openlibrary_service.search_books(q)
    except Exception as exc:
        log.error("Search failed for q=%s type=%s: %s", q, type, exc)
        return html_error(request, f"Search failed: {exc}")

    if not results:
        return HTMLResponse('<p style="color:var(--text-muted);font-size:11px;padding:8px 0;">No results found.</p>')

    # Check which external IDs are already tracked
    external_ids = [r["external_id"] for r in results if r.get("external_id")]
    sources = [r["external_source"] for r in results if r.get("external_source")]

    tracked_ids: set[str] = set()
    if external_ids:
        existing_result = await db.execute(
            select(MediaItem.external_id, UserMedia.id)
            .join(UserMedia, MediaItem.id == UserMedia.media_item_id)
            .where(MediaItem.external_id.in_(external_ids))
        )
        for row in existing_result.all():
            if row.external_id:
                tracked_ids.add(row.external_id)

    # Tag each result with whether it's already tracked
    for r in results:
        r["already_tracked"] = r.get("external_id") in tracked_ids

    return templates.TemplateResponse("partials/search_results.html", {
        "request": request,
        "results": results,
        "search_type": type,
    })


@router.post("/add", response_class=HTMLResponse)
async def add_from_search(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Adds a searched item to the user's list.

    Flow:
      1. Check if media_item already exists by (external_id, external_source).
      2. If not, fetch full details from TMDB/OpenLibrary and create media_item.
      3. Fetch streaming providers (TMDB items only) and store on media_item.
      4. Create user_media record with status = want_to.
      5. Return the new media card partial.

    Idempotent: if user_media already exists, return the existing card.
    """
    form = await request.form()
    external_id = form.get("external_id", "").strip()
    external_source = form.get("external_source", "manual").strip()
    title_hint = form.get("title", "").strip()  # Pre-filled from search result

    if not external_id:
        return html_error(request, "external_id required")

    # ── Step 1: find or create media_item ────────────────────────────────────
    existing = await db.execute(
        select(MediaItem)
        .where(MediaItem.external_id == external_id)
        .where(MediaItem.external_source == external_source)
    )
    item = existing.scalar_one_or_none()

    if not item:
        # ── Step 2: fetch full details ────────────────────────────────────────
        try:
            if external_source == "tmdb_movie":
                details = await tmdb_service.get_movie_details(external_id)
            elif external_source == "tmdb_tv":
                details = await tmdb_service.get_tv_details(external_id)
            elif external_source == "openlibrary":
                details = await openlibrary_service.get_book_details(external_id)
                # Enrich with author if description was fetched
            else:
                details = {
                    "external_id": external_id,
                    "external_source": external_source,
                    "title": title_hint or "Unknown",
                    "media_type": "movie",
                }
        except Exception as exc:
            log.error("Failed to fetch details for %s/%s: %s", external_source, external_id, exc)
            # Fall back to minimal record using search result data
            details = {
                "external_id": external_id,
                "external_source": external_source,
                "title": title_hint or "Unknown",
                "media_type": _infer_media_type(external_source),
            }

        # Map external_source string to enum
        try:
            source_enum = MediaExternalSource(external_source)
        except ValueError:
            source_enum = MediaExternalSource.MANUAL

        try:
            type_enum = MediaType(details.get("media_type", "movie"))
        except ValueError:
            type_enum = MediaType.MOVIE

        item = MediaItem(
            external_id=external_id,
            external_source=source_enum,
            title=details.get("title", title_hint or "Unknown"),
            media_type=type_enum,
            genres=details.get("genres"),
            release_year=details.get("release_year"),
            description=details.get("description"),
            poster_url=details.get("poster_url"),
            external_rating=details.get("external_rating"),
            runtime_minutes=details.get("runtime_minutes"),
            total_seasons=details.get("total_seasons"),
            total_episodes=details.get("total_episodes"),
            author=details.get("author"),
            page_count=details.get("page_count"),
        )
        db.add(item)
        await db.flush()  # Get item.id before Step 3

        # ── Step 3: fetch streaming providers (TMDB only) ─────────────────────
        if external_source in ("tmdb_movie", "tmdb_tv"):
            try:
                provider_ids = await tmdb_service.get_streaming_providers(
                    external_id,
                    media_type=details.get("media_type", "movie"),
                )
                item.streaming_provider_ids = provider_ids if provider_ids else None
                import datetime
                item.streaming_fetched_at = datetime.datetime.utcnow()
            except Exception as exc:
                log.warning("Could not fetch streaming providers for %s: %s", external_id, exc)

        await db.commit()
        await db.refresh(item)

    # ── Step 4: create user_media (idempotent) ────────────────────────────────
    existing_um = await db.execute(
        select(UserMedia).where(UserMedia.media_item_id == item.id)
    )
    um = existing_um.scalar_one_or_none()

    if not um:
        um = UserMedia(media_item_id=item.id, status=UserMediaStatus.WANT_TO)
        db.add(um)
        await db.commit()
        await db.refresh(um)

    return templates.TemplateResponse("partials/media_card.html", {
        "request": request,
        "um": um,
    })


def _infer_media_type(external_source: str) -> str:
    mapping = {
        "tmdb_movie":  "movie",
        "tmdb_tv":     "tv_show",
        "openlibrary": "book",
    }
    return mapping.get(external_source, "movie")
