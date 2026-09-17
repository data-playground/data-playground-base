# domains/soccer/routers/soccer.py
"""
routers/soccer.py — Soccer domain browsing UI (WO#34).

Fixtures/results browsing only for this first pass — see WO#34's Step 1
answer on dashboard/sidebar integration ("future fast-follow"). This
router is registered in main.py and reachable at /soccer, but it is
deliberately NOT wired into routers/dashboard.py's cross-domain summary
or the sidebar's Modules nav in this pass.

Match detail rendering (soccer_match_detail below) uses the real
normalized soccer_match_lineups / soccer_goals / soccer_bookings /
soccer_substitutions / soccer_coaches tables once
parse_finished_match_details() has populated them for a given match —
see lineup_helpers.py for the pitch-token layout math. Before that DAG
task has run for a match (still scheduled, or finished but not yet
parsed this cycle), the template shows an empty state rather than the
old raw-JSON dump.

Stats shown on the match page are deliberately limited to what's
actually confirmed and stored (score, possession, formations). Earlier
WO#34 mockups also showed xG/shots/big-chances/corners/passes/duels/
saves/fouls, but those numbers came from a reference ESPN screenshot
used purely for layout design — never a real FIFA field this pipeline
has parsed. See soccer_match_detail.html for the honest, data-backed
version and a note on which of those could realistically be derived
later from the /timelines event stream.

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
from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, Request, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import select, asc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from core.templating import templates
from domains.soccer.models import (
    SoccerCompetition, SoccerMatch, SoccerMatchLineup,
    SoccerGoal, SoccerBooking, SoccerSubstitution, SoccerCoach,
)
from domains.soccer.lineup_helpers import build_pitch_tokens, build_crest_url

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
    """Single match view — see module docstring for what's real vs. deferred."""
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

    lineup_result = await db.execute(select(SoccerMatchLineup).where(SoccerMatchLineup.match_id == match_id))
    lineups = lineup_result.scalars().all()

    goals_result = await db.execute(
        select(SoccerGoal).where(SoccerGoal.match_id == match_id).order_by(SoccerGoal.minute_numeric)
    )
    goals = goals_result.scalars().all()

    bookings_result = await db.execute(select(SoccerBooking).where(SoccerBooking.match_id == match_id))
    bookings = bookings_result.scalars().all()

    subs_result = await db.execute(
        select(SoccerSubstitution).where(SoccerSubstitution.match_id == match_id).order_by(SoccerSubstitution.minute_numeric)
    )
    subs = subs_result.scalars().all()

    coaches_result = await db.execute(select(SoccerCoach).where(SoccerCoach.match_id == match_id))
    coaches = coaches_result.scalars().all()

    home_lineup = [l for l in lineups if l.team_side == "home"]
    away_lineup = [l for l in lineups if l.team_side == "away"]
    home_goals = [g for g in goals if g.team_side == "home"]
    away_goals = [g for g in goals if g.team_side == "away"]
    home_bookings = [b for b in bookings if b.team_side == "home"]
    away_bookings = [b for b in bookings if b.team_side == "away"]

    # Player-name lookup for the scorer/assist line up top — every
    # player who could possibly score or assist is in soccer_match_lineups
    # (starters and bench alike), so this covers both without a join.
    player_names = {l.fifa_player_id: l.player_name for l in lineups}

    def _scorer_lines(team_goals):
        return [
            {
                "name": player_names.get(g.fifa_player_id, "Unknown"),
                "minute": g.minute_display,
                "assist": player_names.get(g.fifa_assist_player_id) if g.fifa_assist_player_id else None,
            }
            for g in team_goals
        ]

    # role_code 0 = head coach — confirmed pattern on both teams
    # independently for one real match; see SoccerCoach's docstring.
    home_coach = next((c.name for c in coaches if c.team_side == "home" and c.role_code == 0), None)
    away_coach = next((c.name for c in coaches if c.team_side == "away" and c.role_code == 0), None)

    return templates.TemplateResponse("soccer_match_detail.html", {
        "request": request,
        "active_module": "soccer",
        "match": match,
        "has_lineup_data": bool(lineups),
        "home_tokens": build_pitch_tokens(home_lineup, home_goals, home_bookings),
        "away_tokens": build_pitch_tokens(away_lineup, away_goals, away_bookings),
        "home_subs": [s for s in subs if s.team_side == "home"],
        "away_subs": [s for s in subs if s.team_side == "away"],
        "home_scorers": _scorer_lines(home_goals),
        "away_scorers": _scorer_lines(away_goals),
        "home_coach": home_coach,
        "away_coach": away_coach,
        "home_carded_ids": {b.fifa_player_id for b in home_bookings if b.fifa_player_id},
        "away_carded_ids": {b.fifa_player_id for b in away_bookings if b.fifa_player_id},
        "home_crest_url": build_crest_url(match.fifa_home_team_id),
        "away_crest_url": build_crest_url(match.fifa_away_team_id),
    })
