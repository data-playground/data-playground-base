# domains/nba/routers/nba_games.py
"""
NBA domain — v1 dedicated pages: game log + game detail (traditional box
score + play-by-play). See domains/nba/models.py's module docstring for
why the other nine box-score variant tables the DAG ingests have no route
here yet — that's the agreed fast-follow, not an oversight.
"""
import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.templating import templates
from database import get_db
from domains.nba.models import BoxScoreTraditional, Game, Player, PlayByPlayEvent

router = APIRouter(prefix="/nba", tags=["NBA"])


@router.get("", response_class=HTMLResponse)
async def game_log(request: Request, date: str | None = None, db: AsyncSession = Depends(get_db)):
    if date:
        try:
            target_date = datetime.date.fromisoformat(date)
        except ValueError:
            target_date = datetime.date.today()
    else:
        # Default to the most recent date that actually has games, not
        # "today" — on an off-day (All-Star break, start of season, etc.)
        # an empty "today" page is a worse default than the last real slate.
        latest_result = await db.execute(
            select(Game.game_date).order_by(desc(Game.game_date)).limit(1)
        )
        latest = latest_result.scalar_one_or_none()
        target_date = latest or datetime.date.today()

    games_result = await db.execute(
        select(Game)
        .options(selectinload(Game.home_team), selectinload(Game.away_team))
        .where(Game.game_date == target_date)
        .order_by(Game.game_id)
    )
    games = games_result.scalars().all()

    return templates.TemplateResponse("nba_games.html", {
        "request": request,
        "active_module": "nba",
        "target_date": target_date,
        "prev_date": target_date - datetime.timedelta(days=1),
        "next_date": target_date + datetime.timedelta(days=1),
        "games": games,
    })


@router.get("/game/{game_id}", response_class=HTMLResponse)
async def game_detail(request: Request, game_id: str, db: AsyncSession = Depends(get_db)):
    game_result = await db.execute(
        select(Game)
        .options(selectinload(Game.home_team), selectinload(Game.away_team))
        .where(Game.game_id == game_id)
    )
    game = game_result.scalar_one_or_none()
    if game is None:
        return templates.TemplateResponse(
            "404.html",
            {"request": request, "detail": f"No NBA game found for ID {game_id}"},
            status_code=404,
        )

    box_result = await db.execute(
        select(BoxScoreTraditional, Player)
        .outerjoin(Player, Player.person_id == BoxScoreTraditional.person_id)
        .where(BoxScoreTraditional.game_id == game_id)
    )
    box_rows = box_result.all()
    home_box = [(bs, p) for bs, p in box_rows if bs.team_id == game.home_team_id]
    away_box = [(bs, p) for bs, p in box_rows if bs.team_id == game.away_team_id]

    pbp_result = await db.execute(
        select(PlayByPlayEvent)
        .where(PlayByPlayEvent.game_id == game_id)
        .order_by(PlayByPlayEvent.action_number)
    )
    play_by_play = pbp_result.scalars().all()

    return templates.TemplateResponse("nba_game_detail.html", {
        "request": request,
        "active_module": "nba",
        "game": game,
        "home_box": home_box,
        "away_box": away_box,
        "play_by_play": play_by_play,
    })
