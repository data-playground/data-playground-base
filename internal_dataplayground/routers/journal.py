# routers/journal.py
"""
Daily Journal Module

Endpoints:
  GET   /journal                          → Main journal page (today's entry + calendar)
  GET   /journal/{date}                   → Specific date view (YYYY-MM-DD)
  POST  /journal                          → Create or update today's entry
  PATCH /journal/{entry_id}/lock          → Lock an entry (used by DAG + UI)
  GET   /journal/synthesis/latest         → Latest weekly synthesis (JSON)
  GET   /journal/synthesis/{week_start}   → Full synthesis for a week (HTML partial)

Privacy contract: content, gratitude, and challenges are read/written here
but are NEVER forwarded to any external API call anywhere in this router.
"""

import datetime
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import JournalEntry, WeeklySynthesis

log = logging.getLogger(__name__)

router = APIRouter(prefix="/journal", tags=["Journal"])

# Lazy import templates so we can reuse the shared Jinja2 instance
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")

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


async def _get_calendar_dates(db: AsyncSession, days: int = 90) -> set[datetime.date]:
    """Returns the set of dates that have entries, for the last `days` days."""
    cutoff = _today() - datetime.timedelta(days=days)
    result = await db.execute(
        select(JournalEntry.entry_date, JournalEntry.mood_score)
        .where(JournalEntry.entry_date >= cutoff)
        .order_by(JournalEntry.entry_date)
    )
    return {row.entry_date: row.mood_score for row in result.all()}


async def _get_calendar_data(db: AsyncSession, days: int = 90) -> dict:
    """Returns {date: mood_score} for calendar rendering."""
    cutoff = _today() - datetime.timedelta(days=days)
    result = await db.execute(
        select(JournalEntry.entry_date, JournalEntry.mood_score)
        .where(JournalEntry.entry_date >= cutoff)
    )
    return {row.entry_date: row.mood_score for row in result.all()}


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

@router.get("/synthesis/history", response_class=HTMLResponse)
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


@router.get("/synthesis/latest")
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


@router.get("/synthesis/{week_start_date}", response_class=HTMLResponse)
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
        from models import WeeklyPlanDay as _WPD, WeeklyPlan as _WP, WeeklyPlanStatus as _WPS
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


# ── Calendar helpers ───────────────────────────────────────────────────────────

def _build_calendar_months(
    today: datetime.date,
    calendar_data: dict,
) -> list[dict]:
    """
    Returns 3 months of calendar data for the template.
    Each month is a dict with: year, month, month_name, weeks (list of week rows).
    Each day cell: {date, day_num, has_entry, mood_score, is_today, is_future, mood_class}
    """
    months = []
    for month_offset in range(-2, 1):  # two months ago, last month, this month
        # Calculate target month
        m = today.month + month_offset
        y = today.year
        while m <= 0:
            m += 12
            y -= 1
        while m > 12:
            m -= 12
            y += 1

        import calendar as cal_mod
        month_name = datetime.date(y, m, 1).strftime("%B %Y")
        first_day = datetime.date(y, m, 1)
        # weekday() returns 0=Mon, 6=Sun — we want Mon as start
        start_weekday = first_day.weekday()  # 0-6
        num_days = cal_mod.monthrange(y, m)[1]

        weeks = []
        current_week = [None] * start_weekday  # padding for first week
        for day_num in range(1, num_days + 1):
            d = datetime.date(y, m, day_num)
            mood = calendar_data.get(d)
            cell = {
                "date": d,
                "day_num": day_num,
                "has_entry": d in calendar_data,
                "mood_score": mood,
                "is_today": d == today,
                "is_future": d > today,
                "mood_class": _mood_class(mood) if d in calendar_data else "none",
                "date_str": d.isoformat(),
            }
            current_week.append(cell)
            if len(current_week) == 7:
                weeks.append(current_week)
                current_week = []
        if current_week:
            # Pad the last week
            while len(current_week) < 7:
                current_week.append(None)
            weeks.append(current_week)

        months.append({
            "year": y,
            "month": m,
            "month_name": month_name,
            "weeks": weeks,
        })

    return months


def _mood_class(mood_score: Optional[int]) -> str:
    if not mood_score:
        return "none"
    if mood_score <= 2:
        return "low"
    if mood_score == 3:
        return "mid"
    return "high"


async def _calculate_streak(db: AsyncSession, today: datetime.date) -> int:
    """Returns the number of consecutive days with journal entries ending today."""
    result = await db.execute(
        select(JournalEntry.entry_date)
        .where(JournalEntry.entry_date <= today)
        .order_by(desc(JournalEntry.entry_date))
        .limit(365)
    )
    dates = {row.entry_date for row in result.all()}

    streak = 0
    check = today
    while check in dates:
        streak += 1
        check -= datetime.timedelta(days=1)
    return streak
