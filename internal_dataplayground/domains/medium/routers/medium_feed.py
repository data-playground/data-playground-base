# domains/medium/routers/medium_feed.py
"""
The reading page — GET /medium renders tracked articles.

Default view mixes two layouts from the WO#35 mockup review: the most
recent articles get full feed-style rows, everything older renders as
a dense compact list below an "Earlier" divider. Feed-only / card-grid
/ compact-only views are reachable from the same page through a small
button bar — pure client-side visibility toggling, no request.

Filtering (by tracked source) is a plain GET form submit — simplest
thing that works, and it makes the filtered view bookmarkable/
shareable via the URL's ?sources= params.

Pagination is "Show More", not numbered pages, via HTMX
(GET /medium/more). Since all four views render from the same article
list simultaneously (that's what makes switching between them free —
see articles.html), a single "Show More" click has to append the new
batch into all four sections at once, not just the one currently
visible, or switching views after paginating would show stale data.
That's done with HTMX out-of-band swaps (hx-swap-oob) in
partials/more_articles.html — one response, four DOM targets updated.

Offset/limit pagination, not a cursor — simplest thing that works for
a personal app with daily batch inserts. Known, accepted limitation:
if new articles are inserted between "Show More" clicks, the offset
window can shift and produce a skipped or duplicate row at the
boundary. Worth revisiting with a published_at+id cursor if that ever
actually shows up in practice.

ASSUMPTION, not confirmed against base.html: htmx is loaded globally,
inferred from domains/habits/templates/*.html using hx-* attributes
with no local <script src=".../htmx.js"> of its own anywhere in that
domain. If that's wrong, the "Show More" button and OOB swaps below
won't fire — check for a global htmx include in base.html first if so.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from core.templating import templates
from domains.medium.formatting import estimate_read_minutes, relative_time
from domains.medium.models import MediumArticle, MediumFeedSource

router = APIRouter(prefix="/medium", tags=["medium"])

_LATEST_COUNT = 5  # how many articles get full feed-style rows in the default "mixed" view
_PAGE_SIZE = 25    # articles per page — comfortably inside the requested 20-30 range


def _base_query(sources: list[str]):
    query = select(MediumArticle).order_by(MediumArticle.published_at.desc(), MediumArticle.id.desc())
    if sources:
        query = query.where(MediumArticle.source_identifier.in_(sources))
    return query


async def _load_source_options(db: AsyncSession) -> list[dict]:
    """Tracked, active sources for the filter control — not distinct
    values off medium_articles, so a source with zero articles so far
    is still selectable rather than invisible until its first ingest."""
    result = await db.execute(
        select(MediumFeedSource)
        .where(MediumFeedSource.is_active == True)  # noqa: E712
        .order_by(MediumFeedSource.source_type, MediumFeedSource.identifier)
    )
    rows = result.scalars().all()
    return [{"value": row.identifier, "label": row.label or row.identifier} for row in rows]


def _filter_button_label(source_options: list[dict], selected: list[str]) -> str:
    """What the collapsed filter dropdown's toggle button shows."""
    if not selected:
        return "Filter by source"
    if len(selected) == 1:
        match = next((opt["label"] for opt in source_options if opt["value"] == selected[0]), selected[0])
        return match
    return f"{len(selected)} sources selected"


@router.get("")
async def articles_page(
    request: Request,
    sources: list[str] = Query(default=[]),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(_base_query(sources).limit(_PAGE_SIZE))
    rows = result.scalars().all()
    articles = [_present(row) for row in rows]
    has_more = len(rows) == _PAGE_SIZE  # cheap heuristic, not an exact count — see module docstring
    source_options = await _load_source_options(db)

    return templates.TemplateResponse(
        "articles.html",
        {
            "request": request,
            "active_module": "medium",
            "articles": articles,
            "latest": articles[:_LATEST_COUNT],
            "more": articles[_LATEST_COUNT:],
            "source_options": source_options,
            "selected_sources": sources,
            "filter_button_label": _filter_button_label(source_options, sources),
            "has_more": has_more,
            "next_offset": _PAGE_SIZE,
        },
    )


@router.get("/more")
async def more_articles(
    request: Request,
    offset: int = 0,
    sources: list[str] = Query(default=[]),
    db: AsyncSession = Depends(get_db),
):
    """HTMX partial for "Show More" — returns out-of-band swaps for all
    four view sections at once (see module docstring for why)."""
    result = await db.execute(_base_query(sources).offset(offset).limit(_PAGE_SIZE))
    rows = result.scalars().all()
    articles = [_present(row) for row in rows]
    has_more = len(rows) == _PAGE_SIZE

    return templates.TemplateResponse(
        "partials/more_articles.html",
        {
            "request": request,
            "articles": articles,
            "has_more": has_more,
            "next_offset": offset + _PAGE_SIZE,
            "sources": sources,
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
        "thumbnail_url": row.thumbnail_url,
    }
