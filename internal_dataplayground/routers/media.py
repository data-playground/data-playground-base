# routers/media.py
"""
Media Tracker — main board, item CRUD, status and rating management.

Prefix: /media
All page templates extend base.html.
All HTMX partial responses return HTML fragments from templates/partials/.
"""

import datetime
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import (
    MediaItem, UserMedia, TVSeasonProgress, StreamingService,
    UserMediaStatus, MediaType, RecommendationMediaType,
)
from routers._helpers import html_error

router = APIRouter(prefix="/media", tags=["Media"])
templates = Jinja2Templates(directory="templates")
log = logging.getLogger(__name__)


# ── BOARD VIEW ────────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def media_board(
    request: Request,
    db: AsyncSession = Depends(get_db),
    type_filter: Optional[str] = None,      # movie | tv_show | book
    status_filter: Optional[str] = None,   # want_to | in_progress | completed | abandoned
    genre_filter: Optional[str] = None,
    service_filter: Optional[int] = None,  # streaming_services.id
    q: Optional[str] = None,               # title search
):
    """
    Main library view — Netflix-style grid with filter pills at the top.
    All filters are additive (AND). Empty filter = show all.
    """
    # Base query: all user_media with their media_item loaded
    stmt = (
        select(UserMedia)
        .join(MediaItem, UserMedia.media_item_id == MediaItem.id)
        .order_by(UserMedia.updated_at.desc())
    )

    if type_filter and type_filter in [t.value for t in MediaType]:
        stmt = stmt.where(MediaItem.media_type == type_filter)

    if status_filter and status_filter in [s.value for s in UserMediaStatus]:
        stmt = stmt.where(UserMedia.status == status_filter)

    if q:
        stmt = stmt.where(MediaItem.title.ilike(f"%{q}%"))

    if genre_filter:
        stmt = stmt.where(MediaItem.genres.contains(genre_filter))

    result = await db.execute(stmt)
    user_media_list = result.scalars().all()

    # Filter by streaming service (post-query, since streaming_provider_ids is JSON)
    if service_filter:
        svc = await db.get(StreamingService, service_filter)
        if svc and svc.tmdb_provider_id:
            pid = svc.tmdb_provider_id
            user_media_list = [
                um for um in user_media_list
                if pid in (um.media_item.streaming_available_on or [])
            ]

    # Collect unique genres for filter pills
    all_genres: set[str] = set()
    for um in user_media_list:
        for g in (um.media_item.genre_list or []):
            all_genres.add(g)

    # Subscribed streaming services for the filter bar
    svc_result = await db.execute(
        select(StreamingService)
        .where(StreamingService.is_subscribed == True)
        .order_by(StreamingService.sort_order)
    )
    subscribed_services = svc_result.scalars().all()

    # Stats for header
    stats = {
        "total": len(user_media_list),
        "completed": sum(1 for um in user_media_list if um.status == UserMediaStatus.COMPLETED),
        "in_progress": sum(1 for um in user_media_list if um.status == UserMediaStatus.IN_PROGRESS),
        "want_to": sum(1 for um in user_media_list if um.status == UserMediaStatus.WANT_TO),
    }

    return templates.TemplateResponse("media.html", {
        "request": request,
        "active_module": "media",
        "user_media_list": user_media_list,
        "all_genres": sorted(all_genres),
        "subscribed_services": subscribed_services,
        "stats": stats,
        # Active filters (for highlighting pills)
        "type_filter": type_filter,
        "status_filter": status_filter,
        "genre_filter": genre_filter,
        "service_filter": service_filter,
        "q": q or "",
    })


# ── DETAIL DRAWER ─────────────────────────────────────────────────────────────

@router.get("/{user_media_id}/detail", response_class=HTMLResponse)
async def media_detail(
    user_media_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Returns the detail drawer partial for a tracked item."""
    um = await db.get(UserMedia, user_media_id)
    if not um:
        raise HTTPException(status_code=404, detail="Not found")

    # Load streaming services for this item
    provider_ids = um.media_item.streaming_available_on
    streaming_services = []
    if provider_ids:
        svc_result = await db.execute(
            select(StreamingService)
            .where(StreamingService.tmdb_provider_id.in_(provider_ids))
            .order_by(StreamingService.sort_order)
        )
        streaming_services = svc_result.scalars().all()

    from models import PREDEFINED_MOOD_TAGS
    return templates.TemplateResponse("partials/media_detail.html", {
        "request": request,
        "um": um,
        "streaming_services": streaming_services,
        "predefined_mood_tags": PREDEFINED_MOOD_TAGS,
    })


# ── ADD TO LIST ───────────────────────────────────────────────────────────────

@router.post("", response_class=HTMLResponse)
async def add_to_list(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Adds a media_item to the user's list.
    Called from the search results via HTMX — returns an updated card partial.
    Idempotent: if already tracked, returns the existing record.
    """
    form = await request.form()
    media_item_id = int(form.get("media_item_id", 0))
    status_raw = form.get("status", "want_to")

    if not media_item_id:
        return html_error(request, "media_item_id required")

    item = await db.get(MediaItem, media_item_id)
    if not item:
        return html_error(request, "Media item not found", status_code=404)

    try:
        status = UserMediaStatus(status_raw)
    except ValueError:
        status = UserMediaStatus.WANT_TO

    # Check for existing record
    existing = await db.execute(
        select(UserMedia).where(UserMedia.media_item_id == media_item_id)
    )
    um = existing.scalar_one_or_none()

    if not um:
        um = UserMedia(media_item_id=media_item_id, status=status)
        db.add(um)
        await db.commit()
        await db.refresh(um)

    return templates.TemplateResponse("partials/media_card.html", {
        "request": request,
        "um": um,
    })


# ── STATUS UPDATE ─────────────────────────────────────────────────────────────

@router.patch("/{user_media_id}/status", response_class=HTMLResponse)
async def update_status(
    user_media_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Updates the status of a tracked item.
    Returns the updated media card partial.
    """
    form = await request.form()
    status_raw = form.get("status", "")

    um = await db.get(UserMedia, user_media_id)
    if not um:
        return html_error(request, "Not found", status_code=404)

    try:
        um.status = UserMediaStatus(status_raw)
    except ValueError:
        return html_error(request, f"Invalid status: {status_raw}")

    # Auto-set dates
    now = datetime.date.today()
    if um.status == UserMediaStatus.IN_PROGRESS and not um.started_at:
        um.started_at = now
    elif um.status == UserMediaStatus.COMPLETED and not um.completed_at:
        um.completed_at = now

    await db.commit()
    await db.refresh(um)

    return templates.TemplateResponse("partials/media_card.html", {
        "request": request,
        "um": um,
    })


# ── RATING UPDATE ─────────────────────────────────────────────────────────────

@router.patch("/{user_media_id}/rate", response_class=HTMLResponse)
async def update_rating(
    user_media_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Sets user_rating and mood_tags. Returns updated detail drawer rating section.
    """
    form = await request.form()
    rating_raw = form.get("user_rating")
    mood_tags_raw = form.get("mood_tags", "")

    um = await db.get(UserMedia, user_media_id)
    if not um:
        return html_error(request, "Not found", status_code=404)

    if rating_raw is not None:
        try:
            rating = int(rating_raw)
            if 1 <= rating <= 10:
                um.user_rating = rating
        except ValueError:
            pass

    if mood_tags_raw:
        tags = [t.strip() for t in mood_tags_raw.split(",") if t.strip()]
        um.mood_tags = tags if tags else None

    await db.commit()
    await db.refresh(um)

    return templates.TemplateResponse("partials/media_rating.html", {
        "request": request,
        "um": um,
    })


# ── NOTES UPDATE ──────────────────────────────────────────────────────────────

@router.patch("/{user_media_id}/notes", response_class=HTMLResponse)
async def update_notes(
    user_media_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Updates personal notes. Returns a simple confirmation fragment."""
    form = await request.form()
    um = await db.get(UserMedia, user_media_id)
    if not um:
        return html_error(request, "Not found", status_code=404)

    um.notes = form.get("notes") or None
    await db.commit()
    return HTMLResponse('<span style="color:var(--green);font-size:10px;">✓ Saved</span>')


# ── TV SEASON PROGRESS ────────────────────────────────────────────────────────

@router.post("/{user_media_id}/seasons/{season_number}", response_class=HTMLResponse)
async def update_season_progress(
    user_media_id: int,
    season_number: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Updates episode progress for a specific TV season.
    Creates the season row if it doesn't exist (sparse tracking).
    """
    form = await request.form()
    episodes_watched = int(form.get("episodes_watched", 0))
    total_episodes_raw = form.get("total_episodes")
    total_episodes = int(total_episodes_raw) if total_episodes_raw else None

    um = await db.get(UserMedia, user_media_id)
    if not um:
        return html_error(request, "Not found", status_code=404)

    # Upsert season progress row
    existing = await db.execute(
        select(TVSeasonProgress)
        .where(TVSeasonProgress.user_media_id == user_media_id)
        .where(TVSeasonProgress.season_number == season_number)
    )
    sp = existing.scalar_one_or_none()

    if sp:
        sp.episodes_watched = episodes_watched
        if total_episodes is not None:
            sp.total_episodes = total_episodes
    else:
        sp = TVSeasonProgress(
            user_media_id=user_media_id,
            season_number=season_number,
            episodes_watched=episodes_watched,
            total_episodes=total_episodes,
        )
        db.add(sp)

    await db.commit()
    await db.refresh(um)

    return templates.TemplateResponse("partials/media_seasons.html", {
        "request": request,
        "um": um,
    })


# ── REMOVE FROM LIST ──────────────────────────────────────────────────────────

@router.delete("/{user_media_id}", response_class=HTMLResponse)
async def remove_from_list(
    user_media_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Removes an item from the user's list.
    Keeps the media_item record (it may appear in recommendations).
    Returns an empty response so HTMX removes the card from the DOM.
    """
    um = await db.get(UserMedia, user_media_id)
    if not um:
        raise HTTPException(status_code=404, detail="Not found")

    await db.delete(um)
    await db.commit()
    return HTMLResponse("")  # HTMX swap removes the card
