# routers/journal.py
"""
Daily Journal Module

Endpoints:
  GET   /journal                          → Main journal page (today's entry + calendar)
  GET   /journal/{date}                   → Specific date view (YYYY-MM-DD)
  POST  /journal                          → Create or update today's entry
  PATCH /journal/{entry_id}/lock          → Lock an entry (used by DAG + UI)

Weekly-synthesis endpoints (GET /journal/synthesis/*) moved to
routers/journal_synthesis.py in WO#25 — see that file's docstring.
Calendar/streak helpers moved to routers/_calendar.py in WO#25.

Privacy contract: content, gratitude, and challenges are read/written here
but are NEVER forwarded to any external API call anywhere in this router.
"""

import datetime
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from sqlalchemy import select, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.journal.models import JournalEntry, WeeklySynthesis
from domains.journal.routers._calendar import (
    _build_calendar_months,
    _calculate_streak,
    _get_calendar_data,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/journal", tags=["Journal"])

from core.templating import templates

# Maximum days in the past a user can create a new entry (no arbitrary backdating)
MAX_BACKDATE_DAYS = 7


# ── Helpers ────────────────────────────────────────────────────────────────────

def _today() -> datetime.date:
    return datetime.date.today()


async def _get_entry_by_date(
    db: AsyncSession, entry_date: datetime.date
) -> Optional[JournalEntry]:
    result = await db.execute(
        select(JournalEntry).where(JournalEntry.entry_date == entry_date)
    )
    return result.scalar_one_or_none()


# ── Main journal page ──────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def journal_home(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    today = _today()
    today_entry = await _get_entry_by_date(db, today)

    # Auto-lock today's entry if it's past 24 hours and not yet locked
    if today_entry and today_entry.should_be_locked and not today_entry.is_locked:
        today_entry.is_locked = True
        await db.commit()
        await db.refresh(today_entry)

    # Last 12 weekly syntheses for the history panel
    syntheses_result = await db.execute(
        select(WeeklySynthesis)
        .order_by(desc(WeeklySynthesis.week_start_date))
        .limit(12)
    )
    recent_syntheses = syntheses_result.scalars().all()

    # Latest synthesis for the dashboard card
    latest_synthesis = recent_syntheses[0] if recent_syntheses else None

    # Calendar data: {date: mood_score} for the last 90 days
    calendar_data = await _get_calendar_data(db, days=90)

    # Build 3-month calendar structure
    calendar_months = _build_calendar_months(today, calendar_data)

    # Streak calculation
    streak = await _calculate_streak(db, today)

    return templates.TemplateResponse("journal.html", {
        "request": request,
        "today_entry": today_entry,
        "today": today,
        "recent_syntheses": recent_syntheses,
        "latest_synthesis": latest_synthesis,
        "calendar_months": calendar_months,
        "calendar_data": calendar_data,
        "streak": streak,
        "active_module": "journal",
        "max_backdate_days": MAX_BACKDATE_DAYS,
    })


# ── Specific date view ─────────────────────────────────────────────────────────

@router.get("/{date_str}", response_class=HTMLResponse)
async def journal_date(
    date_str: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    today = _today()
    try:
        entry_date = datetime.date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=422, detail="Date must be YYYY-MM-DD")

    # Reject future dates
    if entry_date > today:
        raise HTTPException(status_code=400, detail="Cannot journal for future dates")

    # Reject dates too far in the past (no entry exists and too old to create)
    days_ago = (today - entry_date).days
    entry = await _get_entry_by_date(db, entry_date)

    if not entry and days_ago > MAX_BACKDATE_DAYS:
        return templates.TemplateResponse("partials/journal_no_entry.html", {
            "request": request,
            "entry_date": entry_date,
            "today": today,
        })

    # Auto-lock if past 24h
    if entry and entry.should_be_locked and not entry.is_locked:
        entry.is_locked = True
        await db.commit()
        await db.refresh(entry)

    calendar_data = await _get_calendar_data(db, days=90)
    calendar_months = _build_calendar_months(today, calendar_data)
    streak = await _calculate_streak(db, today)

    syntheses_result = await db.execute(
        select(WeeklySynthesis)
        .order_by(desc(WeeklySynthesis.week_start_date))
        .limit(12)
    )
    recent_syntheses = syntheses_result.scalars().all()
    latest_synthesis = recent_syntheses[0] if recent_syntheses else None

    return templates.TemplateResponse("journal.html", {
        "request": request,
        "today_entry": entry,
        "today": today,
        "viewing_date": entry_date,
        "is_past_date": entry_date < today,
        "recent_syntheses": recent_syntheses,
        "latest_synthesis": latest_synthesis,
        "calendar_months": calendar_months,
        "calendar_data": calendar_data,
        "streak": streak,
        "active_module": "journal",
        "max_backdate_days": MAX_BACKDATE_DAYS,
    })


# ── Create / update entry ──────────────────────────────────────────────────────

@router.post("", response_class=HTMLResponse)
async def save_entry(
    request: Request,
    mood_score: Optional[int] = Form(None),
    energy_score: Optional[int] = Form(None),
    content: Optional[str] = Form(None),
    gratitude: Optional[str] = Form(None),
    challenges: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    today = _today()
    existing = await _get_entry_by_date(db, today)

    if existing and existing.is_locked:
        return templates.TemplateResponse("partials/journal_entry_saved.html", {
            "request": request,
            "error": "This entry is locked and can no longer be edited.",
            "entry": existing,
        }, status_code=403)

    # Validate score ranges
    def _clamp_score(val: Optional[int]) -> Optional[int]:
        if val is None:
            return None
        return max(1, min(5, val))

    mood = _clamp_score(mood_score)
    energy = _clamp_score(energy_score)

    # PRIVACY: content, gratitude, challenges are stored locally only.
    # They are intentionally NOT forwarded to any external service.
    clean_content = (content or "").strip() or None
    clean_gratitude = (gratitude or "").strip() or None
    clean_challenges = (challenges or "").strip() or None

    if existing:
        existing.mood_score = mood
        existing.energy_score = energy
        existing.content = clean_content
        existing.gratitude = clean_gratitude
        existing.challenges = clean_challenges
        existing.updated_at = datetime.datetime.utcnow()
        entry = existing
    else:
        entry = JournalEntry(
            entry_date=today,
            mood_score=mood,
            energy_score=energy,
            content=clean_content,
            gratitude=clean_gratitude,
            challenges=clean_challenges,
        )
        db.add(entry)

    await db.commit()
    await db.refresh(entry)

    try:
        from sqlalchemy import select as _select
        from domains.planning.models import WeeklyPlanDay as _WPD, WeeklyPlan as _WP, WeeklyPlanStatus as _WPS
        plan_day_result = await db.execute(
            _select(_WPD)
            .join(_WP, _WPD.weekly_plan_id == _WP.id)
            .where(_WPD.plan_date == today)
            .where(_WP.status.in_([_WPS.CONFIRMED, _WPS.ACTIVE]))
            .limit(1)
        )
        plan_day = plan_day_result.scalar_one_or_none()
        if plan_day and not plan_day.journal_entry_id:
            plan_day.journal_entry_id = entry.id
            await db.commit()
    except Exception:
        pass  # Don't fail the journal save if this linking fails

    return templates.TemplateResponse("partials/journal_entry_saved.html", {
        "request": request,
        "entry": entry,
        "error": None,
    })


# ── Lock endpoint ──────────────────────────────────────────────────────────────

@router.patch("/{entry_id}/lock", response_class=HTMLResponse)
async def lock_entry(
    entry_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    entry = await db.get(JournalEntry, entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    entry.is_locked = True
    await db.commit()

    return HTMLResponse(
        f'<span class="lock-badge locked">🔒 Locked</span>',
        status_code=200,
    )
