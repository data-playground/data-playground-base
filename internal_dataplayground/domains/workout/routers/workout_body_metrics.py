# domains/workout/routers/workout_body_metrics.py
"""
Workout Tracker — Body Metrics

WO#29 Part C (sign-off obtained): split out of workout_log.py, which used
to export both `router` (/workout/sessions/*) and this module's
`body_metrics_router` (/workout/body-metrics/*) from a single file. WO#8's
own HARD BOUNDARIES said not to merge or further split that structure
during that migration — this split explicitly reverses that precedent,
with the project owner's sign-off, because workout_log.py had grown to
357 lines. Both routers are still registered as separate
app.include_router() calls in main.py, exactly as before.

Endpoints:
  POST   /workout/body-metrics                 → Log body weight (upsert)
  GET    /workout/body-metrics                 → Last 90 days of body metrics (JSON)
"""

import datetime
import logging
from decimal import Decimal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from core.templating import templates
from domains.workout.models import BodyMetric
from domains.workout.routers._shared import parse_weight_unit

log = logging.getLogger(__name__)

# Second, distinct APIRouter — was module-level `body_metrics_router` in
# workout_log.py before this split. Name kept identical so main.py's
# `workout_log.body_metrics_router` reference just becomes
# `workout_body_metrics.body_metrics_router`.
body_metrics_router = APIRouter(prefix="/workout/body-metrics", tags=["Workout"])


@body_metrics_router.post("", response_class=HTMLResponse)
async def log_body_metric(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Upserts a body metric entry for a given date (defaults to today).
    Returns the updated sparkline data as JSON for the chart.
    """
    form = await request.form()
    date_raw = str(form.get("metric_date", "")).strip()
    try:
        metric_date = datetime.date.fromisoformat(date_raw) if date_raw else datetime.date.today()
    except ValueError:
        metric_date = datetime.date.today()

    weight_raw = str(form.get("weight", "")).strip()
    weight = Decimal(weight_raw) if weight_raw else None

    weight_unit_raw = str(form.get("weight_unit", "lb")).strip()
    weight_unit = parse_weight_unit(weight_unit_raw)

    bf_raw = str(form.get("body_fat_pct", "")).strip()
    body_fat_pct = Decimal(bf_raw) if bf_raw else None

    notes_raw = str(form.get("notes", "")).strip()

    existing_result = await db.execute(
        select(BodyMetric).where(BodyMetric.metric_date == metric_date)
    )
    existing = existing_result.scalar_one_or_none()

    if existing:
        if weight is not None:
            existing.weight = weight
            existing.weight_unit = weight_unit
        if body_fat_pct is not None:
            existing.body_fat_pct = body_fat_pct
        if notes_raw:
            existing.notes = notes_raw
        metric = existing
    else:
        metric = BodyMetric(
            metric_date=metric_date,
            weight=weight,
            weight_unit=weight_unit,
            body_fat_pct=body_fat_pct,
            notes=notes_raw or None,
        )
        db.add(metric)

    await db.commit()
    await db.refresh(metric)

    return templates.TemplateResponse("partials/workout/body_metric_saved.html", {
        "request": request,
        "metric": metric,
    })


@body_metrics_router.get("", response_class=JSONResponse)
async def get_body_metrics(db: AsyncSession = Depends(get_db)):
    """Returns last 90 days of body metrics as JSON for chart rendering."""
    cutoff = datetime.date.today() - datetime.timedelta(days=90)
    result = await db.execute(
        select(BodyMetric)
        .where(BodyMetric.metric_date >= cutoff)
        .order_by(BodyMetric.metric_date)
    )
    metrics = result.scalars().all()
    return [
        {
            "date": str(m.metric_date),
            "weight": float(m.weight) if m.weight else None,
            "unit": m.weight_unit.value,
            "body_fat_pct": float(m.body_fat_pct) if m.body_fat_pct else None,
        }
        for m in metrics
    ]
