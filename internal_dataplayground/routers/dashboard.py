# routers/dashboard.py
"""
Life OS — Hub Dashboard

Aggregates data from every module into a single at-a-glance view:
  - ATS pipeline snapshot
  - Top unapplied high-fit jobs
  - Fit score distribution histogram
  - Finance summary (current month)
  - Blog pipeline counts
  - Staging queue status
"""

import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func, desc, extract, case, not_, exists
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import (
    Job, ApplicationLog, ApplicationStatus,
    StagingJob, StagingJobStatus,
    Transaction, BlogIdea, BlogIdeaStatus,
)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])
templates = Jinja2Templates(directory="templates")


@router.get("", response_class=HTMLResponse)
async def dashboard(request: Request, db: AsyncSession = Depends(get_db)):

    today = datetime.date.today()

    # ── 1. Top unapplied jobs ─────────────────────────────────────────────────
    # Jobs that have NO application log entry (never touched)
    applied_job_ids_subq = select(ApplicationLog.job_id).distinct().scalar_subquery()

    top_jobs_result = await db.execute(
        select(Job)
        .where(Job.fit_score >= 85)
        .where(Job.ID.not_in(applied_job_ids_subq))
        .order_by(desc(Job.fit_score))
        .limit(6)
    )
    top_jobs = top_jobs_result.scalars().all()

    # ── 2. Fit score histogram ────────────────────────────────────────────────
    # Count jobs per score bucket
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

    # Ordered buckets for rendering
    bucket_order = ["95-100", "90-94", "85-89", "80-84", "75-79", "70-74", "60-69", "50-59", "<50"]
    histogram = [(b, hist_raw.get(b, 0)) for b in bucket_order]
    hist_max = max((c for _, c in histogram), default=1)

    # ── 3. ATS pipeline snapshot ──────────────────────────────────────────────
    # Most recent status per job using a subquery
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

    # Ordered pipeline stages for the funnel display
    pipeline_stages = [
        (ApplicationStatus.APPLIED,              "Applied",     "var(--accent)"),
        (ApplicationStatus.PHONE_SCREEN,         "Phone",       "var(--blue)"),
        (ApplicationStatus.INTERVIEWING,         "Interview",   "var(--yellow)"),
        (ApplicationStatus.TECHNICAL_ASSESSMENT, "Technical",   "var(--yellow)"),
        (ApplicationStatus.OFFER,                "Offer",       "var(--green)"),
        (ApplicationStatus.REJECTED,             "Rejected",    "var(--red)"),
        (ApplicationStatus.CLOSED,               "Closed",      "var(--text-muted)"),
    ]
    pipeline = [
        {"label": label, "count": pipeline_raw.get(status, 0), "color": color}
        for status, label, color in pipeline_stages
    ]
    total_in_pipeline = sum(p["count"] for p in pipeline)

    # ── 4. Finance summary (current month) ───────────────────────────────────
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

    # ── 5. Blog pipeline counts ───────────────────────────────────────────────
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
    blog_active = sum(
        blog_raw.get(s, 0) for s in [
            BlogIdeaStatus.WRITING_IN_PROGRESS,
            BlogIdeaStatus.WAITING_FOR_REVIEW,
            BlogIdeaStatus.REVIEW_COMPLETED,
        ]
    )
    blog_ready = blog_raw.get(BlogIdeaStatus.READY_TO_PUBLISH, 0)
    blog_total = sum(blog_raw.values())

    # ── 6. Staging queue ──────────────────────────────────────────────────────
    staging_result = await db.execute(
        select(StagingJob.status, func.count(StagingJob.id).label("count"))
        .group_by(StagingJob.status)
    )
    staging_raw = {row.status: row.count for row in staging_result.all()}
    staging_pending    = staging_raw.get(StagingJobStatus.PENDING, 0)
    staging_processing = staging_raw.get(StagingJobStatus.PROCESSING, 0)
    staging_done       = staging_raw.get(StagingJobStatus.DONE, 0)

    # ── 7. Total job counts ───────────────────────────────────────────────────
    total_jobs_result = await db.execute(select(func.count(Job.ID)))
    total_jobs = total_jobs_result.scalar() or 0

    high_fit_result = await db.execute(
        select(func.count(Job.ID)).where(Job.fit_score >= 90)
    )
    high_fit_count = high_fit_result.scalar() or 0

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "active_module": "dashboard",
        # Jobs
        "top_jobs": top_jobs,
        "total_jobs": total_jobs,
        "high_fit_count": high_fit_count,
        # Histogram
        "histogram": histogram,
        "hist_max": hist_max,
        # Pipeline
        "pipeline": pipeline,
        "total_in_pipeline": total_in_pipeline,
        # Finance
        "fin_income": fin_income,
        "fin_expenses": fin_expenses,
        "fin_net": fin_net,
        "has_finance": has_finance,
        "today": today,
        # Blog
        "blog_backlog": blog_backlog,
        "blog_active": blog_active,
        "blog_ready": blog_ready,
        "blog_total": blog_total,
        # Staging
        "staging_pending": staging_pending,
        "staging_processing": staging_processing,
        "staging_done": staging_done,
    })
