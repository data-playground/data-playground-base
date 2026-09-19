# routers/_shared.py
"""
Shared helpers for the Habit Tracker routers.

Consumed by habits.py (daily check-in surface) and habits_settings.py
(settings / CRUD surface). Nothing in this file is a route handler — no
@router decorators live here. Moved verbatim from habits.py (WO#23), mirroring
the precedent in domains/workout/routers/_shared.py.

Streak algorithm:
  Walks backwards from yesterday. Counts consecutive days where a log exists,
  allowing up to grace_period_days missed days within the window before
  breaking the streak. Today is excluded (the day is not over yet).

Sort order:
  Incomplete habits render first (ascending sort_order), completed habits
  render last (ascending sort_order within completed group). This is a
  display sort applied in the router — not a DB sort.
"""

import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from domains.habits.models import Habit, HabitLog, HabitSettings


async def _get_grace_period(db: AsyncSession) -> int:
    """Reads grace_period_days from the single habit_settings row."""
    result = await db.execute(select(HabitSettings).limit(1))
    settings = result.scalar_one_or_none()
    return settings.grace_period_days if settings else 1


async def _get_logged_dates_for_habit(
    db: AsyncSession, habit_id: int, days: int = 400
) -> set[datetime.date]:
    """
    Returns a set of logged dates for a habit over the last `days` days.
    Used by both the streak calculation and the heatmap endpoint.
    """
    cutoff = datetime.date.today() - datetime.timedelta(days=days)
    result = await db.execute(
        select(HabitLog.logged_date)
        .where(HabitLog.habit_id == habit_id)
        .where(HabitLog.logged_date >= cutoff)
    )
    return {row[0] for row in result.all()}


def _calculate_streak(logged_dates: set[datetime.date], grace_period: int) -> int:
    """
    Counts consecutive logged days walking backwards from yesterday.

    Grace period: up to `grace_period` consecutive missed days are tolerated
    before the streak breaks. Setting grace_period=0 requires strict
    consecutive days.

    Today is intentionally excluded — the day isn't over yet, and including
    it would make the streak feel inflated before the user has actually
    completed their habits.

    Args:
        logged_dates: Set of dates that have a log entry.
        grace_period: Number of consecutive misses allowed before streak breaks.

    Returns:
        Integer streak count (0 if yesterday was not logged and grace=0).
    """
    if not logged_dates:
        return 0

    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    streak = 0
    missed_in_window = 0
    cursor = yesterday

    # Walk backwards day by day for up to 2 years (safety cap)
    for _ in range(730):
        if cursor in logged_dates:
            streak += 1
            missed_in_window = 0  # reset miss counter on a logged day
        else:
            missed_in_window += 1
            if missed_in_window > grace_period:
                break
            # Within grace window — keep walking but don't increment streak
        cursor -= datetime.timedelta(days=1)

    return streak


async def _build_habit_view(
    db: AsyncSession,
    habit: Habit,
    today: datetime.date,
    grace_period: int,
    logged_today_ids: set[int],
) -> dict:
    """
    Attaches today_logged and streak to a habit for template rendering.
    Returns a plain dict so it can be used both in the full page and partials.
    """
    logged_dates = await _get_logged_dates_for_habit(db, habit.id, days=400)
    streak = _calculate_streak(logged_dates, grace_period)
    return {
        "habit": habit,
        "today_logged": habit.id in logged_today_ids,
        "streak": streak,
    }


async def _get_today_logged_ids(db: AsyncSession, today: datetime.date) -> set[int]:
    """Returns the set of habit_ids that have been logged today."""
    result = await db.execute(
        select(HabitLog.habit_id).where(HabitLog.logged_date == today)
    )
    return {row[0] for row in result.all()}


def _sort_habits_for_display(habit_views: list[dict]) -> list[dict]:
    """
    Incomplete habits first (ascending sort_order), completed habits last
    (ascending sort_order within completed group). This is the display rule:
    checking off a habit moves it to the bottom of the page.
    """
    incomplete = [h for h in habit_views if not h["today_logged"]]
    complete = [h for h in habit_views if h["today_logged"]]
    incomplete.sort(key=lambda h: h["habit"].sort_order)
    complete.sort(key=lambda h: h["habit"].sort_order)
    return incomplete + complete


def _get_week_dates() -> list[datetime.date]:
    """Returns the 7 dates of the current week, starting Sunday."""
    today = datetime.date.today()
    # weekday(): Monday=0, Sunday=6
    days_since_sunday = (today.weekday() + 1) % 7
    sunday = today - datetime.timedelta(days=days_since_sunday)
    return [sunday + datetime.timedelta(days=i) for i in range(7)]
