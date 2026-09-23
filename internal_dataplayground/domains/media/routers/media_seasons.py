# routers/media_seasons.py
"""
Media Tracker — TV season progress tracking.

Split out of media.py (WO#28 Part B) to keep that router under the
300-line limit. Pure relocation — no behavior change, no URL change.
Still mounted under the /media prefix, so the endpoint's full path is
unchanged: POST /media/{user_media_id}/seasons/{season_number}.
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.media.models import TVSeasonProgress, UserMedia
from routers._helpers import html_error
from core.templating import templates

router = APIRouter(prefix="/media", tags=["Media Seasons"])


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
