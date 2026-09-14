# domains/medium/routers/medium_feed.py
"""
The reading page — GET /medium renders tracked articles.

Default view mixes two layouts from the WO#35 mockup review: the most
recent articles get full feed-style rows, everything older renders as
a dense compact list below an "Earlier" divider. Feed-only / card-grid
/ compact-only views are reachable from the same page through a small
button bar. All four views render from one query result — the button
bar just toggles which section is visible client-side, so switching
is instant with no reload and no extra request.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.medium.formatting import estimate_read_minutes, relative_time
from domains.medium.models import MediumArticle

router = APIRouter(prefix="/medium", tags=["medium"])
templates = Jinja2Templates(directory="templates")

_LATEST_COUNT = 5  # how many articles get full feed-style rows in the default "mixed" view
_PAGE_LIMIT = 60   # articles pulled per request — no pagination yet, revisit once volume warrants it


@router.get("")
async def articles_page(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(MediumArticle).order_by(MediumArticle.published_at.desc()).limit(_PAGE_LIMIT)
    )
    rows = result.scalars().all()

    articles = [_present(row) for row in rows]
    return templates.TemplateResponse(
        "medium/articles.html",
        {
            "request": request,
            "articles": articles,
            "latest": articles[:_LATEST_COUNT],
            "more": articles[_LATEST_COUNT:],
        },
    )


def _present(row: MediumArticle) -> dict:
    """Flattens a MediumArticle row into template-ready display fields —
    keeps relative-time/read-time formatting out of Jinja."""
    return {
        "title": row.title,
        "url": row.url,
        "author": row.author,
        "source_label": row.source_label or row.source_identifier,
        "published_relative": relative_time(row.published_at) if row.published_at else "",
        "read_minutes": estimate_read_minutes(row.content_html or row.summary),
        "tags": row.tags or [],
    }
