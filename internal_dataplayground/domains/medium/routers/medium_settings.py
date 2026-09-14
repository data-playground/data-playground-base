# domains/medium/routers/medium_settings.py
"""
Lets you manage the tracked-sources list (profiles, publications,
topics, custom domains) from a page in the app instead of editing code.
This IS the "manual recommendations" workflow in practice: browse
medium.com's recommended feed yourself while logged in in a real
browser, and when something looks worth following, add it here instead
of touching a Python file.

Two ways to add a source:
  - Auto-detect (POST /sources/detect): paste the plain Medium link you'd
    normally visit, and it's classified + added via the on-demand
    life_os_medium_detect_source Airflow DAG — not inline in this
    request. That's deliberate: Airflow's own run history is the
    "what happened and can I retry it" story here, not anything this
    router tracks itself. Concretely, this means the response to
    submitting the form is only ever "queued" or "couldn't reach
    Airflow" — never "found 3 articles, added as publication X". If
    detection itself succeeds or fails, that shows up as a new row
    below (or doesn't) and in Airflow's task log, not as a banner here.
  - Manual (POST /sources): pick the type from a dropdown yourself,
    inserted directly, no Airflow involved — kept as a fallback for
    whatever candidate_sources() doesn't recognize (see its own
    docstring for known gaps), and it's a pure local DB write so
    there's nothing for it to fail asynchronously on.

Duplicate detection is NOT checked here for the auto-detect path,
unlike the manual one — this router doesn't know the final
(source_type, identifier) until identify_source() actually runs, and
that now happens inside the DAG, not in this process. A resubmission
of an already-tracked source fails there instead, on
medium_feed_sources' own unique constraint — surfaces as a failed
Airflow task with a clear "Duplicate entry" error, not a banner here.

Everything else — list/toggle/delete — only ever talks to this app's
own database. _preview_url() calls build_feed_url(), which is pure
string formatting, not a network request.
"""
from __future__ import annotations

import logging
from urllib.parse import quote

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from core.templating import templates
from domains.medium.models import MediumFeedSource
from domains.medium.rss_ingest import FeedSource as RssFeedSource
from domains.medium.rss_ingest import FeedSourceType, build_feed_url
from services.airflow_service import trigger_airflow

log = logging.getLogger(__name__)

router = APIRouter(prefix="/medium/settings", tags=["medium-settings"])
# templates = Jinja2Templates(directory="templates")


def _preview_url(row: MediumFeedSource) -> str | None:
    try:
        return build_feed_url(RssFeedSource(type=FeedSourceType(row.source_type), identifier=row.identifier))
    except ValueError:
        return None


def _banner_from_query(request: Request) -> dict | None:
    """Turns the redirect-carried query params from /sources/detect into
    a template-ready banner. Query-param flash messages rather than
    sessions/cookies — simplest thing that works for a single-user app.

    Only two states now that detection itself runs in Airflow, not
    here: the trigger call to Airflow either succeeded (queued) or
    couldn't be made at all (trigger_error). Whether detection itself
    then succeeds or fails is Airflow's story, not this function's."""
    qp = request.query_params

    if qp.get("queued"):
        return {
            "kind": "success",
            "message": (
                f'Queued detection for "{qp.get("value", "")}" (Airflow run {qp.get("run_id", "")}). '
                "Check Airflow for the result — success shows up as a new row below; "
                "failure shows up as a failed task with the reason in its log."
            ),
        }
    if qp.get("trigger_error"):
        return {
            "kind": "error",
            "message": (
                f'Could not reach Airflow to queue detection for "{qp.get("value", "")}". '
                "Check the Airflow webserver, then try again."
            ),
        }
    return None


@router.get("")
async def settings_page(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(MediumFeedSource).order_by(MediumFeedSource.source_type, MediumFeedSource.identifier)
    )
    rows = result.scalars().all()
    sources = [{"row": row, "feed_url": _preview_url(row)} for row in rows]
    return templates.TemplateResponse(
        "medium/settings.html",
        {
            "request": request,
            "sources": sources,
            "source_types": [t.value for t in FeedSourceType],
            "banner": _banner_from_query(request),
        },
    )


@router.post("/sources/detect")
async def detect_and_add_source(
    medium_url: str = Form(...),
    label: str | None = Form(None),
):
    # No db dependency here at all now — this endpoint's only job is
    # firing the DAG. It doesn't know source_type/identifier (that's
    # what the DAG figures out) so there's nothing of its own to write.
    try:
        run_id = await trigger_airflow(
            "life_os_medium_detect_source",
            conf={"raw_url": medium_url, "label": label},
        )
    except httpx.HTTPError as exc:
        log.error("Failed to trigger life_os_medium_detect_source for %r: %s", medium_url, exc)
        return RedirectResponse(
            url=f"/medium/settings?trigger_error=1&value={quote(medium_url)}",
            status_code=303,
        )

    return RedirectResponse(
        url=f"/medium/settings?queued=1&value={quote(medium_url)}&run_id={quote(run_id)}",
        status_code=303,
    )


@router.post("/sources")
async def add_source(
    source_type: FeedSourceType = Form(...),
    identifier: str = Form(...),
    label: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    identifier = identifier.strip()
    if source_type is FeedSourceType.PROFILE:
        identifier = identifier.lstrip("@")
    if not identifier:
        raise HTTPException(status_code=400, detail="Identifier cannot be empty")

    existing = await db.execute(
        select(MediumFeedSource).where(
            MediumFeedSource.source_type == source_type.value,
            MediumFeedSource.identifier == identifier,
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail="That source is already tracked")

    db.add(MediumFeedSource(source_type=source_type.value, identifier=identifier, label=label or None))
    await db.commit()
    return RedirectResponse(url="/medium/settings", status_code=303)


@router.post("/sources/{source_id}/toggle")
async def toggle_source(source_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(MediumFeedSource).where(MediumFeedSource.id == source_id))
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Source not found")
    row.is_active = not row.is_active
    await db.commit()
    return RedirectResponse(url="/medium/settings", status_code=303)


@router.post("/sources/{source_id}/delete")
async def delete_source(source_id: int, db: AsyncSession = Depends(get_db)):
    await db.execute(delete(MediumFeedSource).where(MediumFeedSource.id == source_id))
    await db.commit()
    return RedirectResponse(url="/medium/settings", status_code=303)
