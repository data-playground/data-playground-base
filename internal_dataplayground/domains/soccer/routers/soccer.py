# domains/soccer/routers/soccer.py
"""
routers/soccer.py — Soccer domain browsing UI (WO#34).

Fixtures/results browsing only for this first pass — see WO#34's Step 1
answer on dashboard/sidebar integration ("future fast-follow"). This
router is registered in main.py and reachable at /soccer, but it is
deliberately NOT wired into routers/dashboard.py's cross-domain summary
or the sidebar's Modules nav in this pass.

Match-level detail rendering reads straight from soccer_raw_payloads
(see domains/soccer/models.py's module docstring on why match_details/
match_events aren't normalized yet) rather than a dedicated ORM model —
the payload is displayed as formatted JSON rather than parsed into a
template-friendly shape. This is a known limitation to revisit once
FIFA's field names are verified against a live response (see
airflow/agents/soccer_agents.py's FIELD-NAME CAVEAT).

`since` defaults to yesterday: competitions can carry years of backfilled
history (World Cup 2022 onward, for example), and a fixtures list with no
default date floor would be dominated by that archive rather than
anything actually relevant right now. `until` defaults to tomorrow for
the same reason on the future side — with matches backfilled and
scheduled potentially years out, an unbounded upper edge is rarely what
someone actually wants to see by default. "Show full history" (in
soccer.html) explicitly overrides BOTH bounds to a wide sentinel range
rather than omitting them — since omitting them now just falls back to
the yesterday/tomorrow defaults instead of "no filter."
"""
import json
from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, Request, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import select, asc, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from core.templating import templates
from domains.soccer.models import SoccerCompetition, SoccerMatch, SoccerRawPayload

router = APIRouter(prefix="/soccer", tags=["Soccer"])


@router.get("", response_class=HTMLResponse)
async def soccer_home(
    request: Request,
    db: AsyncSession = Depends(get_db),
    competition_id: int | None = Query(default=None),
    status: str | None = Query(default=None, description="scheduled | live | postponed | finished"),
    since: str | None = Query(default=None, description="YYYY-MM-DD — only show matches from this date forward. Defaults to yesterday."),
    until: str | None = Query(default=None, description="YYYY-MM-DD — only show matches up to and including this date. Defaults to tomorrow."),
):
    """Fixtures/results list, filterable by competition, status, and a date range (defaults to yesterday through tomorrow)."""
    if since:
        try:
            since_date = date.fromisoformat(since)
        except ValueError:
            since_date = date.today() - timedelta(days=1)
    else:
        since_date = date.today() - timedelta(days=1)

    if until:
        try:
            until_date = date.fromisoformat(until)
        except ValueError:
            until_date = date.today() + timedelta(days=1)
    else:
        until_date = date.today() + timedelta(days=1)

    comp_result = await db.execute(
        select(SoccerCompetition)
        .where(SoccerCompetition.is_active.is_(True))
        .order_by(SoccerCompetition.name)
    )
    competitions = comp_result.scalars().all()

    # Ascending, not descending — once the default view is "yesterday
    # onward" this reads as an actual upcoming-fixtures list (soonest
    # first) rather than surfacing the single furthest-out future match.
    query = select(SoccerMatch).order_by(asc(SoccerMatch.kickoff_at))
    if competition_id:
        query = query.where(SoccerMatch.competition_id == competition_id)
    if status:
        query = query.where(SoccerMatch.status_label == status)
    query = query.where(
        (SoccerMatch.kickoff_at >= datetime.combine(since_date, datetime.min.time()))
        | (SoccerMatch.kickoff_at.is_(None))
    )
    # end-of-day bound so a match kicking off later on until_date itself
    # is still included, not excluded by an exact-midnight cutoff
    query = query.where(
        (SoccerMatch.kickoff_at <= datetime.combine(until_date, datetime.max.time()))
        | (SoccerMatch.kickoff_at.is_(None))
    )
    query = query.limit(100)

    match_result = await db.execute(query)
    matches = match_result.scalars().all()

    return templates.TemplateResponse("soccer.html", {
        "request": request,
        "active_module": "soccer",
        "competitions": competitions,
        "matches": matches,
        "selected_competition_id": competition_id,
        "selected_status": status,
        "since": since_date.isoformat(),
        "until": until_date.isoformat(),
    })


@router.get("/match/{match_id}", response_class=HTMLResponse)
async def soccer_match_detail(request: Request, match_id: int, db: AsyncSession = Depends(get_db)):
    """
    Single match view. Shows normalized summary fields plus the most
    recently fetched raw /live and /timelines payloads, formatted as
    JSON — see module docstring on why these aren't parsed further yet.
    """
    match_result = await db.execute(select(SoccerMatch).where(SoccerMatch.id == match_id))
    match = match_result.scalar_one_or_none()
    if not match:
        return templates.TemplateResponse(
            "soccer.html",
            {
                "request": request, "active_module": "soccer",
                "competitions": [], "matches": [],
                "selected_competition_id": None, "selected_status": None,
                "since": "", "until": "", "error": "Match not found.",
            },
            status_code=404,
        )

    details_result = await db.execute(
        select(SoccerRawPayload)
        .where(SoccerRawPayload.endpoint == "match_details")
        .where(SoccerRawPayload.fifa_match_id == match.fifa_match_id)
        .order_by(desc(SoccerRawPayload.fetched_at))
        .limit(1)
    )
    details_row = details_result.scalar_one_or_none()

    events_result = await db.execute(
        select(SoccerRawPayload)
        .where(SoccerRawPayload.endpoint == "match_events")
        .where(SoccerRawPayload.fifa_match_id == match.fifa_match_id)
        .order_by(desc(SoccerRawPayload.fetched_at))
        .limit(1)
    )
    events_row = events_result.scalar_one_or_none()

    return templates.TemplateResponse("soccer_match_detail.html", {
        "request": request,
        "active_module": "soccer",
        "match": match,
        "details_json": json.dumps(details_row.payload, indent=2) if details_row else None,
        "events_json": json.dumps(events_row.payload, indent=2) if events_row else None,
    })
