# routers/_calendar.py
"""
Calendar & streak helpers for the Daily Journal module.

Pure data-shaping functions used by journal.py to render the 3-month
calendar grid, mood dots, and streak count on the main journal page.

Split out of journal.py in WO#25 purely to bring journal.py under the
300-line router limit (CONTRIBUTING.md / GOVERNANCE.md §1.2) — no behavior
change. These functions have no endpoints of their own and are not
registered with any APIRouter.

None of these functions touch JournalEntry.content, .gratitude, or
.challenges — only entry_date and the numeric mood_score — so the
content/gratitude/challenges privacy contract documented in journal.py
does not apply to this file.
"""

import datetime
from typing import Optional

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from domains.journal.models import JournalEntry


def _today() -> datetime.date:
    """Local copy of journal.py's `_today()`.

    Duplicated rather than imported from journal.py: journal.py imports
    these calendar helpers, so importing `_today` back from journal.py
    would create a circular import. `_today()` wasn't in WO#25's SCOPE
    list of functions to move, so it stays defined in journal.py too —
    this is a deliberate one-line duplication, not a divergence in
    behavior (both are `return datetime.date.today()`). See WO#25 report.
    """
    return datetime.date.today()


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
