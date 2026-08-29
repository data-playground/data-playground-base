# routers/dashboard.py
"""
Life OS — Hub Dashboard

Aggregates data from every module into a single at-a-glance view.
"""

import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, desc, extract, case, not_, exists
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.jobs.models import Job, ApplicationLog, ApplicationStatus, StagingJob, StagingJobStatus
from domains.finance.models import Transaction
from domains.blog.models import BlogIdea, BlogIdeaStatus
from domains.habits.models import Habit, HabitLog, HabitSettings
from domains.journal.models import JournalEntry, WeeklySynthesis

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])
templates = Jinja2Templates(directory="templates")


@router.get("", response_class=HTMLResponse)
async def dashboard(request: Request, db: AsyncSession = Depends(get_db)):

    today = datetime.date.today()

    # ── Top unapplied jobs ────────────────────────────────────────────────────
    applied_job_ids_subq = select(ApplicationLog.job_id).distinct().scalar_subquery()

    top_jobs_result = await db.execute(
        select(Job)
        .where(Job.fit_score >= 85)
        .where(Job.ID.not_in(applied_job_ids_subq))
        .order_by(desc(Job.fit_score))
        .limit(6)
    )
    top_jobs = top_jobs_result.scalars().all()

    # ── Fit score histogram ───────────────────────────────────────────────────
    hist_result = await db.execute(
        select(
            case(
                (Job.fit_score >= 95, "95-100"),
                (Job.fit_score >= 90, "90-94"),
                (Job.fit_score >= 85, "85-89"),
                (Job.fit_score >= 80, "80-84"),
                (Job.fit_score >= 75, "75-79"),
                (Job.fit_score >= 70, "70-74"),
                (Job.fit_score >= 60, "60-69"),
                (Job.fit_score >= 50, "50-59"),
                else_="<50"
            ).label("bucket"),
            func.count(Job.ID).label("count"),
        )
        .group_by("bucket")
    )
    hist_raw = {row.bucket: row.count for row in hist_result.all()}
    bucket_order = ["95-100", "90-94", "85-89", "80-84", "75-79", "70-74", "60-69", "50-59", "<50"]
    histogram = [(b, hist_raw.get(b, 0)) for b in bucket_order]
    hist_max = max((c for _, c in histogram), default=1)

    # ── ATS pipeline snapshot ─────────────────────────────────────────────────
    latest_log_subq = (
        select(
            ApplicationLog.job_id,
            func.max(ApplicationLog.created_at).label("max_ts"),
        )
        .group_by(ApplicationLog.job_id)
        .subquery()
    )
    pipeline_result = await db.execute(
        select(ApplicationLog.status, func.count(ApplicationLog.id).label("count"))
        .join(
            latest_log_subq,
            (ApplicationLog.job_id == latest_log_subq.c.job_id)
            & (ApplicationLog.created_at == latest_log_subq.c.max_ts),
        )
        .group_by(ApplicationLog.status)
    )
    pipeline_raw = {row.status: row.count for row in pipeline_result.all()}

    pipeline_stages = [
        (ApplicationStatus.APPLIED,              "Applied",   "var(--accent)"),
        (ApplicationStatus.PHONE_SCREEN,         "Phone",     "var(--blue)"),
        (ApplicationStatus.INTERVIEWING,         "Interview", "var(--yellow)"),
        (ApplicationStatus.TECHNICAL_ASSESSMENT, "Technical", "var(--yellow)"),
        (ApplicationStatus.OFFER,                "Offer",     "var(--green)"),
        (ApplicationStatus.REJECTED,             "Rejected",  "var(--red)"),
        (ApplicationStatus.CLOSED,               "Closed",    "var(--text-muted)"),
    ]
    pipeline = [
        {"label": label, "count": pipeline_raw.get(status, 0), "color": color}
        for status, label, color in pipeline_stages
    ]
    total_in_pipeline = sum(p["count"] for p in pipeline)

    # ── Finance summary ───────────────────────────────────────────────────────
    fin_result = await db.execute(
        select(
            func.sum(case((Transaction.amount > 0, Transaction.amount), else_=0)).label("income"),
            func.sum(case((Transaction.amount < 0, Transaction.amount), else_=0)).label("expenses"),
        )
        .where(extract("month", Transaction.date) == today.month)
        .where(extract("year", Transaction.date) == today.year)
    )
    fin_row = fin_result.one()
    fin_income   = float(fin_row.income   or 0)
    fin_expenses = float(fin_row.expenses or 0)
    fin_net      = fin_income + fin_expenses
    has_finance  = (fin_income != 0 or fin_expenses != 0)

    # ── Blog pipeline counts ──────────────────────────────────────────────────
    blog_result = await db.execute(
        select(BlogIdea.status, func.count(BlogIdea.id).label("count"))
        .group_by(BlogIdea.status)
    )
    blog_raw = {row.status: row.count for row in blog_result.all()}

    blog_backlog = sum(
        blog_raw.get(s, 0) for s in [
            BlogIdeaStatus.IDEA_GENERATED,
            BlogIdeaStatus.WAITING_FOR_WRITING_TRIGGER,
        ]
    )
    # in_development is active work — counts alongside writing/review
    blog_active = sum(
        blog_raw.get(s, 0) for s in [
            BlogIdeaStatus.IN_DEVELOPMENT,
            BlogIdeaStatus.WRITING_IN_PROGRESS,
            BlogIdeaStatus.WAITING_FOR_REVIEW,
            BlogIdeaStatus.REVIEW_COMPLETED,
        ]
    )
    blog_ready = blog_raw.get(BlogIdeaStatus.READY_TO_PUBLISH, 0)
    blog_total = sum(blog_raw.values())

    # ── Staging queue ─────────────────────────────────────────────────────────
    staging_result = await db.execute(
        select(StagingJob.status, func.count(StagingJob.id).label("count"))
        .group_by(StagingJob.status)
    )
    staging_raw = {row.status: row.count for row in staging_result.all()}
    staging_pending    = staging_raw.get(StagingJobStatus.PENDING, 0)
    staging_processing = staging_raw.get(StagingJobStatus.PROCESSING, 0)
    staging_done       = staging_raw.get(StagingJobStatus.DONE, 0)

    # ── Job counts ────────────────────────────────────────────────────────────
    total_jobs_result = await db.execute(select(func.count(Job.ID)))
    total_jobs = total_jobs_result.scalar() or 0

    high_fit_result = await db.execute(
        select(func.count(Job.ID)).where(Job.fit_score >= 90)
    )
    high_fit_count = high_fit_result.scalar() or 0
    
    
    # ── 8. Habit summary ──────────────────────────────────────────────────
    _today_date = datetime.date.today()
 
    habit_count_result = await db.execute(
        select(func.count(Habit.id)).where(Habit.is_active == True)
    )
    habits_total = habit_count_result.scalar() or 0
 
    habit_today_result = await db.execute(
        select(func.count(HabitLog.id))
        .where(HabitLog.logged_date == _today_date)
    )
    habits_done_today = habit_today_result.scalar() or 0
 
    grace_result = await db.execute(select(HabitSettings).limit(1))
    grace_settings = grace_result.scalar_one_or_none()
    grace_period = grace_settings.grace_period_days if grace_settings else 1
 
    active_habits_result = await db.execute(
        select(Habit).where(Habit.is_active == True)
    )
    active_habits = active_habits_result.scalars().all()
 
    cutoff = _today_date - datetime.timedelta(days=400)
    all_logs_result = await db.execute(
        select(HabitLog.habit_id, HabitLog.logged_date)
        .where(HabitLog.logged_date >= cutoff)
    )
    logs_by_habit: dict[int, set] = {}
    for _row in all_logs_result.all():
        logs_by_habit.setdefault(_row[0], set()).add(_row[1])
 
    def _dash_streak(dates: set) -> int:
        if not dates:
            return 0
        yesterday = _today_date - datetime.timedelta(days=1)
        streak = 0
        missed = 0
        cursor = yesterday
        for _ in range(730):
            if cursor in dates:
                streak += 1
                missed = 0
            else:
                missed += 1
                if missed > grace_period:
                    break
            cursor -= datetime.timedelta(days=1)
        return streak
 
    best_streak = max(
        (_dash_streak(logs_by_habit.get(h.id, set())) for h in active_habits),
        default=0,
    )
 
    week_start = _today_date - datetime.timedelta(days=6)
    week_logs_result = await db.execute(
        select(func.count(HabitLog.id))
        .where(HabitLog.logged_date >= week_start)
        .where(HabitLog.logged_date <= _today_date)
    )
    week_logs_count = week_logs_result.scalar() or 0
    possible = habits_total * 7
    week_completion_pct = round((week_logs_count / possible) * 100) if possible > 0 else 0
    
    # ── 9. Journal — today's entry + streak + latest synthesis ───────────────────

    import datetime as _dt

    _today = _dt.date.today()

    # Today's entry (mood + energy only — not the text fields)
    today_journal_result = await db.execute(
        select(JournalEntry.mood_score, JournalEntry.energy_score, JournalEntry.is_locked)
        .where(JournalEntry.entry_date == _today)
    )
    today_journal_row = today_journal_result.one_or_none()
    today_mood = today_journal_row.mood_score if today_journal_row else None
    today_energy = today_journal_row.energy_score if today_journal_row else None
    has_today_journal = today_journal_row is not None

    # Journal streak (consecutive days with entries)
    streak_result = await db.execute(
        select(JournalEntry.entry_date)
        .where(JournalEntry.entry_date <= _today)
        .order_by(desc(JournalEntry.entry_date))
        .limit(365)
    )
    streak_dates = {row.entry_date for row in streak_result.all()}
    journal_streak = 0
    _check = _today
    while _check in streak_dates:
        journal_streak += 1
        _check -= _dt.timedelta(days=1)

    # Latest weekly synthesis
    latest_synthesis_result = await db.execute(
        select(WeeklySynthesis)
        .order_by(desc(WeeklySynthesis.week_start_date))
        .limit(1)
    )
    latest_synthesis = latest_synthesis_result.scalar_one_or_none()

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "active_module": "dashboard",
        "top_jobs": top_jobs,
        "total_jobs": total_jobs,
        "high_fit_count": high_fit_count,
        "histogram": histogram,
        "hist_max": hist_max,
        "pipeline": pipeline,
        "total_in_pipeline": total_in_pipeline,
        "fin_income": fin_income,
        "fin_expenses": fin_expenses,
        "fin_net": fin_net,
        "has_finance": has_finance,
        "today": today,
        "blog_backlog": blog_backlog,
        "blog_active": blog_active,
        "blog_ready": blog_ready,
        "blog_total": blog_total,
        "staging_pending": staging_pending,
        "staging_processing": staging_processing,
        "staging_done": staging_done,
        "habits_total": habits_total,
        "habits_done_today": habits_done_today,
        "best_streak": best_streak,
        "week_completion_pct": week_completion_pct,
        "today_mood": today_mood,
        "today_energy": today_energy,
        "has_today_journal": has_today_journal,
        "journal_streak": journal_streak,
        "latest_synthesis": latest_synthesis,
    })
