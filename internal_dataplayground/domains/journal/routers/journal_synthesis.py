# routers/journal_synthesis.py
"""
Weekly Synthesis endpoints for the Daily Journal module.

Endpoints:
  GET /journal/synthesis/history         → Full synthesis history (HTML page)
  GET /journal/synthesis/latest          → Latest weekly synthesis (JSON)
  GET /journal/synthesis/{week_start}    → Full synthesis for a week (HTML partial)

Split out of journal.py in WO#25 to bring journal.py under the 300-line
router limit — no behavior change. This was already an implied boundary:
these three endpoints are the only ones that render journal_synthesis.html
and partials/synthesis_detail.html.

Registered as its own APIRouter (prefix="/journal/synthesis") rather than
sharing journal.py's router object, matching the specific-router-before-
catch-all pattern already used elsewhere in this codebase (e.g. the
workout, media, and recipes domains' multi-router setups in main.py).

PRIVACY NOTE: WeeklySynthesis rows are generated exclusively from numeric
mood_score/energy_score aggregates (see domains/journal/models.py). This
router never reads JournalEntry.content, .gratitude, or .challenges — the
fields journal.py's docstring documents as never forwarded externally —
so the privacy contract in journal.py does not apply to (and is not
duplicated in) this file.
"""

import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.journal.models import WeeklySynthesis

router = APIRouter(prefix="/journal/synthesis", tags=["Journal"])

from core.templating import templates


# ── Synthesis history page ──────────────────────────────────────────────────────

@router.get("/history", response_class=HTMLResponse)
async def synthesis_history(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(WeeklySynthesis)
        .order_by(desc(WeeklySynthesis.week_start_date))
    )
    syntheses = result.scalars().all()
    return templates.TemplateResponse("journal_synthesis.html", {
        "request": request,
        "syntheses": syntheses,
        "active_module": "journal_synthesis",
    })


# ── Latest synthesis (JSON, consumed by the dashboard) ──────────────────────────

@router.get("/latest")
async def latest_synthesis_json(db: AsyncSession = Depends(get_db)):
    """JSON endpoint consumed by the dashboard."""
    result = await db.execute(
        select(WeeklySynthesis)
        .order_by(desc(WeeklySynthesis.week_start_date))
        .limit(1)
    )
    synthesis = result.scalar_one_or_none()
    if not synthesis:
        return JSONResponse({"synthesis": None})
    return JSONResponse({
        "synthesis": {
            "id": synthesis.id,
            "week_label": synthesis.week_label,
            "avg_mood": float(synthesis.avg_mood) if synthesis.avg_mood else None,
            "avg_energy": float(synthesis.avg_energy) if synthesis.avg_energy else None,
            "synthesis_text": synthesis.synthesis_text,
            "generated_at": synthesis.generated_at.isoformat(),
        }
    })


# ── Full synthesis for a given week (HTML partial, loaded into the drawer) ──────

@router.get("/{week_start_date}", response_class=HTMLResponse)
async def synthesis_detail(
    week_start_date: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        week_date = datetime.date.fromisoformat(week_start_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="Date must be YYYY-MM-DD")

    result = await db.execute(
        select(WeeklySynthesis).where(WeeklySynthesis.week_start_date == week_date)
    )
    synthesis = result.scalar_one_or_none()
    if not synthesis:
        raise HTTPException(status_code=404, detail="Synthesis not found for that week")

    return templates.TemplateResponse("partials/synthesis_detail.html", {
        "request": request,
        "synthesis": synthesis,
    })
