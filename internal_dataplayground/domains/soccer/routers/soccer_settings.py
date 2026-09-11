# domains/soccer/routers/soccer_settings.py
"""
routers/soccer_settings.py — Soccer domain settings & watch-list admin
(WO#34 fast-follow).

Lets the user:
  - Tune the daily ingest DAG's rolling window (window_past_days /
    window_future_days), stored in soccer_settings and read by
    airflow/dags/soccer/life_os_soccer_ingest.py::_get_window_days() at
    the start of each run (falling back to that DAG's own module-level
    defaults if no row exists yet).
  - Manage the soccer_competitions watch list: toggle a competition
    active/inactive, remove one that has no ingested matches yet, and
    search FIFA's live competitions list to add a new one — this plays
    the same "confirm before saving" admin role
    services/ats_slug_service.py plays for the jobs domain.

fetch_competitions() lives in airflow/agents/soccer_agents.py (DAG-side
code), but importing it from FastAPI-side code is fine and has
precedent: services/recipe_service.py already imports
airflow.agents.recipe_agents.agent_normalize_ingredients the same way.
The DAG/FastAPI boundary rule in CONTRIBUTING.md / GOVERNANCE.md §2.2 is
one-directional — DAGs never import routers/services/models — nothing
forbids a router importing a DAG-safe agent module.

Kept as its own router file (rather than folded into routers/soccer.py)
per GOVERNANCE §1.2's router-splitting guidance — this is a distinct
concern (admin/config vs. browsing) and keeps both files well under the
300-line ceiling.
"""
from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import date

from database import get_db
from core.templating import templates
from routers._helpers import html_error
from domains.soccer.models import SoccerCompetition, SoccerMatch, SoccerSettings
from airflow.agents.soccer_agents import fetch_competitions

router = APIRouter(prefix="/soccer/settings", tags=["Soccer Settings"])


async def _get_or_create_settings(db: AsyncSession) -> SoccerSettings:
    """Singleton settings row — same get-or-default shape as HabitSettings elsewhere in this app."""
    result = await db.execute(select(SoccerSettings).limit(1))
    settings = result.scalar_one_or_none()
    if not settings:
        settings = SoccerSettings()
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
    return settings


async def _render_watchlist_rows(request: Request, db: AsyncSession, status_code: int = 200) -> HTMLResponse:
    """
    Shared re-render of the watch-list table body, used by every
    mutating endpoint below — including their error paths. This is
    deliberate: every one of those endpoints targets #watchlist-body (a
    <tbody>), and html_error()'s <div>-based fragment isn't valid content
    there. Returning this same partial with a non-2xx status_code instead
    keeps the table well-formed either way; watchlist_rows.html's
    hx-on:htmx:after-request checks event.detail.successful to pick the
    right toast wording.
    """
    comp_result = await db.execute(select(SoccerCompetition).order_by(SoccerCompetition.name))
    competitions = comp_result.scalars().all()
    return templates.TemplateResponse(
        "partials/watchlist_rows.html",
        {"request": request, "competitions": competitions},
        status_code=status_code,
    )


@router.get("", response_class=HTMLResponse)
async def soccer_settings_home(request: Request, db: AsyncSession = Depends(get_db)):
    settings = await _get_or_create_settings(db)

    comp_result = await db.execute(select(SoccerCompetition).order_by(SoccerCompetition.name))
    competitions = comp_result.scalars().all()

    return templates.TemplateResponse("soccer_settings.html", {
        "request": request,
        "active_module": "soccer",
        "settings": settings,
        "competitions": competitions,
    })


@router.post("/window")
async def update_window(
    request: Request,
    db: AsyncSession = Depends(get_db),
    window_past_days: int = Form(...),
    window_future_days: int = Form(...),
):
    if window_past_days < 0 or window_future_days < 1:
        return html_error(
            request,
            "Past days must be 0 or more, and future days must be at least 1.",
        )

    settings = await _get_or_create_settings(db)
    settings.window_past_days = window_past_days
    settings.window_future_days = window_future_days
    await db.commit()

    return RedirectResponse(url="/soccer/settings?msg=Window+updated", status_code=303)


@router.post("/competitions/{competition_id}/toggle", response_class=HTMLResponse)
async def toggle_competition(request: Request, competition_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(SoccerCompetition).where(SoccerCompetition.id == competition_id))
    comp = result.scalar_one_or_none()
    if not comp:
        return await _render_watchlist_rows(request, db, status_code=404)

    comp.is_active = not comp.is_active
    await db.commit()

    return await _render_watchlist_rows(request, db)


@router.post("/competitions/{competition_id}/delete", response_class=HTMLResponse)
async def delete_competition(request: Request, competition_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(SoccerCompetition).where(SoccerCompetition.id == competition_id))
    comp = result.scalar_one_or_none()
    if not comp:
        return await _render_watchlist_rows(request, db, status_code=404)

    match_count_result = await db.execute(
        select(func.count(SoccerMatch.id)).where(SoccerMatch.competition_id == competition_id)
    )
    if (match_count_result.scalar() or 0) > 0:
        # 409, not html_error — this response still has to be valid
        # <tbody> content (see _render_watchlist_rows' docstring). The
        # "can't remove, has matches" explanation surfaces via the
        # non-2xx status driving the client-side toast instead.
        return await _render_watchlist_rows(request, db, status_code=409)

    await db.delete(comp)
    await db.commit()

    return await _render_watchlist_rows(request, db)


@router.post("/competitions/add", response_class=HTMLResponse)
async def add_competition(
    request: Request,
    db: AsyncSession = Depends(get_db),
    fifa_competition_id: str = Form(...),
    name: str = Form(...),
    backfill_from_date: str = Form(default=""),
):
    """
    Adds a competition to the watch list. backfill_from_date is optional
    and only meaningful once, right now — see SoccerCompetition's model
    docstring: it only affects ingest_fixtures() the very first time this
    competition has zero matches on file. Leaving it blank means the
    competition starts tracking from the ordinary rolling window (recent
    past + near-term future) with no history pull. This field used to be
    a hardcoded per-competition value in the DAG's own seed list — moved
    here so backfill depth is a per-competition choice made when adding
    it, not a code change.
    """
    existing = await db.execute(
        select(SoccerCompetition).where(SoccerCompetition.fifa_competition_id == fifa_competition_id)
    )
    if existing.scalar_one_or_none():
        return await _render_watchlist_rows(request, db, status_code=409)

    parsed_backfill = None
    if backfill_from_date:
        try:
            parsed_backfill = date.fromisoformat(backfill_from_date)
        except ValueError:
            return await _render_watchlist_rows(request, db, status_code=422)

    db.add(SoccerCompetition(
        fifa_competition_id=fifa_competition_id,
        name=name,
        is_active=True,
        backfill_from_date=parsed_backfill,
    ))
    await db.commit()

    return await _render_watchlist_rows(request, db)


@router.get("/competitions/search", response_class=HTMLResponse)
async def search_fifa_competitions(request: Request, db: AsyncSession = Depends(get_db), q: str = ""):
    """
    Live-fetches FIFA's full competitions list (cached — see
    soccer_agents.py) and filters it by `q`. This is the "quick button to
    fetch the competition list from FIFA" — each result not already in
    the watch list gets an Add button.
    """
    try:
        all_competitions = fetch_competitions()
    except Exception as exc:
        return html_error(request, f"Couldn't reach FIFA's competitions list: {exc}")

    if q:
        q_lower = q.lower()
        all_competitions = [c for c in all_competitions if q_lower in (c.get("name") or "").lower()]

    # FIFA's own list runs 200+ entries — cap what an empty/short query
    # dumps into the page. A real search term narrows this naturally.
    all_competitions = all_competitions[:40]

    watched_result = await db.execute(select(SoccerCompetition.fifa_competition_id))
    watched_ids = {row[0] for row in watched_result.all()}

    return templates.TemplateResponse("partials/competition_search_results.html", {
        "request": request,
        "results": all_competitions,
        "watched_ids": watched_ids,
        "query": q,
    })
