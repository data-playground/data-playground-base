# routers/staging.py
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from database import get_db
from models import StagingJob, StagingJobStatus
from routers._helpers import html_error

router = APIRouter(prefix="/jobs", tags=["Staging"])
templates = Jinja2Templates(directory="templates")


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
