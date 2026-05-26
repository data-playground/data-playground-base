# routers/media.py
"""
Media Tracker — main board, item CRUD, status and rating management.
Prefix: /media
"""

import datetime
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import (
    MediaItem, UserMedia, TVSeasonProgress, StreamingService,
    UserMediaStatus, MediaType, RecommendationMediaType, PREDEFINED_MOOD_TAGS,
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
    type_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    genre_filter: Optional[str] = None,
    service_filter: Optional[int] = None,
    q: Optional[str] = None,
):
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

    result = await db.execute(stmt)
    user_media_list = result.scalars().all()

    if service_filter:
        svc = await db.get(StreamingService, service_filter)
        if svc and svc.tmdb_provider_id:
            pid = svc.tmdb_provider_id
            user_media_list = [
                um for um in user_media_list
                if pid in (um.media_item.streaming_available_on or [])
            ]

    all_genres: set[str] = set()
    for um in user_media_list:
        for g in (um.media_item.genre_list or []):
            all_genres.add(g)

    svc_result = await db.execute(
        select(StreamingService)
        .where(StreamingService.is_subscribed == True)
        .order_by(StreamingService.sort_order)
    )
    subscribed_services = svc_result.scalars().all()

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
    um = await db.get(UserMedia, user_media_id)
    if not um:
        raise HTTPException(status_code=404, detail="Not found")

    provider_ids = um.media_item.streaming_available_on
    streaming_services = []
    if provider_ids:
        svc_result = await db.execute(
            select(StreamingService)
            .where(StreamingService.tmdb_provider_id.in_(provider_ids))
            .order_by(StreamingService.sort_order)
        )
        streaming_services = svc_result.scalars().all()

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
    Updates status. Returns different partials depending on context:
    - If called from the grid card (hx-target=#media-card-N) → returns media_card.html
    - If called from the detail drawer (hx-target=#drawer-status-N) → returns drawer_status.html
    The router always returns the drawer status block; the card self-updates
    via a separate JS refresh call triggered from the drawer.
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

    now = datetime.date.today()
    if um.status == UserMediaStatus.IN_PROGRESS and not um.started_at:
        um.started_at = now
    elif um.status == UserMediaStatus.COMPLETED and not um.completed_at:
        um.completed_at = now

    await db.commit()
    await db.refresh(um)

    # Check what's being targeted to return the right partial
    target = request.headers.get("hx-target", "")
    if f"media-card-{user_media_id}" in target:
        # Called from grid card quick-action buttons
        return templates.TemplateResponse("partials/media_card.html", {
            "request": request, "um": um,
        })

    # Called from detail drawer — return updated status buttons block
    return templates.TemplateResponse("partials/media_drawer_status.html", {
        "request": request, "um": um,
    })


# ── RATING UPDATE ─────────────────────────────────────────────────────────────

@router.patch("/{user_media_id}/rate", response_class=HTMLResponse)
async def update_rating(
    user_media_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
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
        "request": request, "um": um,
    })


# ── NOTES UPDATE ──────────────────────────────────────────────────────────────

@router.patch("/{user_media_id}/notes", response_class=HTMLResponse)
async def update_notes(
    user_media_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
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
    form = await request.form()
    episodes_watched = int(form.get("episodes_watched", 0))
    total_episodes_raw = form.get("total_episodes")
    total_episodes = int(total_episodes_raw) if total_episodes_raw else None

    um = await db.get(UserMedia, user_media_id)
    if not um:
        return html_error(request, "Not found", status_code=404)

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
        "request": request, "um": um,
    })


# ── REMOVE FROM LIST ──────────────────────────────────────────────────────────

@router.delete("/{user_media_id}", response_class=HTMLResponse)
async def remove_from_list(
    user_media_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    um = await db.get(UserMedia, user_media_id)
    if not um:
        raise HTTPException(status_code=404, detail="Not found")
    await db.delete(um)
    await db.commit()
    return HTMLResponse("")
