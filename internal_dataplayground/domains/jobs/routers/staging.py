# domains/jobs/routers/staging.py
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from database import get_db
from domains.jobs.models import StagingJob, StagingJobStatus
from routers._helpers import html_error
from services.airflow_service import trigger_airflow

from core.templating import templates

router = APIRouter(prefix="/jobs", tags=["Staging"])


@router.post("/stage", response_class=HTMLResponse)
async def stage_job(request: Request, db: AsyncSession = Depends(get_db)):
    form = await request.form()
    job_link = str(form.get("job_link", "")).strip()
    job_search = str(form.get("job_search", "")).strip() or None

    if not job_link or "linkedin.com/jobs" not in job_link:
        return html_error(request, "Please paste a valid LinkedIn job URL.", status_code=422)

    existing = await db.execute(
        select(StagingJob)
        .where(StagingJob.job_link == job_link)
        .where(StagingJob.status.in_([StagingJobStatus.PENDING, StagingJobStatus.PROCESSING]))
    )
    if existing.scalars().first():
        return html_error(request, "This job is already in the queue.", status_code=409)

    entry = StagingJob(job_link=job_link, job_search=job_search)
    db.add(entry)
    await db.commit()
    await db.refresh(entry)

    return templates.TemplateResponse(
        "partials/staging_row.html",
        {"request": request, "job": entry},
    )


@router.get("/stage", response_class=HTMLResponse)
async def list_staged_jobs(request: Request, db: AsyncSession = Depends(get_db)):
    stmt = (
        select(StagingJob)
        .order_by(
            desc(StagingJob.status.in_([StagingJobStatus.PENDING, StagingJobStatus.PROCESSING])),
            desc(StagingJob.created_at),
        )
        .limit(50)
    )
    result = await db.execute(stmt)
    staged = result.scalars().all()
    return templates.TemplateResponse(
        "partials/staging_queue.html",
        {"request": request, "staged": staged},
    )


@router.post("/stage/process", response_class=HTMLResponse)
async def process_staging_queue(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Manually triggers the Staging Promoter DAG (life_os_staging_promoter)
    immediately instead of waiting for its daily schedule — wired to the
    "⚡ Process Now" button next to the staging queue.

    Triggers with no conf, so the DAG itself decides what's PENDING at run
    time. Checks for PENDING rows first purely to give the person clicking
    the button honest feedback — clicking with an empty queue still
    triggers the DAG (harmless no-op on its end) but says so rather than
    implying something is about to happen.

    Both the "nothing pending" and "Airflow unreachable" paths below return
    HTTP 200 (via html_error's status_code param) rather than a 4xx/5xx,
    specifically so this button — wired via hx-post, not the raw-fetch
    pattern stage_job's JS uses — is guaranteed to swap the message into
    view regardless of htmx's error-response-swap configuration, which
    lives in base.html/base.js and wasn't available to check here. The
    message text alone communicates success vs. failure.
    """
    pending_count_result = await db.execute(
        select(func.count(StagingJob.id)).where(StagingJob.status == StagingJobStatus.PENDING)
    )
    pending_count = pending_count_result.scalar() or 0

    if pending_count == 0:
        return html_error(
            request,
            "Nothing PENDING in the queue right now — nothing to process.",
            status_code=200,
        )

    try:
        run_id = await trigger_airflow("life_os_staging_promoter")
    except Exception:
        # 200, not an error status — see note above stage_job/process_staging_queue
        # about htmx's default non-2xx swap behavior; the message text alone
        # communicates the failure here rather than relying on status code.
        return html_error(
            request,
            "Couldn't reach Airflow to trigger processing. It will still run on its "
            "next scheduled pass.",
            status_code=200,
        )

    return templates.TemplateResponse(
        "partials/staging_process_feedback.html",
        {"request": request, "run_id": run_id, "pending_count": pending_count},
    )
