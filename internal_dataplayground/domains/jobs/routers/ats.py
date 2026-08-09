# domains/jobs/routers/ats.py
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from database import get_db
from domains.jobs.models import (
    ApplicationLog, ApplicationLogCreate, ApplicationLogResponse, ApplicationStatus, Job,
)

from core.templating import templates

router = APIRouter(prefix="/ats", tags=["ATS"])


@router.post("/log", response_class=HTMLResponse)
async def create_application_log(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Log a new status change for a job application.
    Accepts form data from HTMX hx-vals and returns an HTML fragment
    of the updated ATS button row so HTMX can swap it in place.
    """
    form = await request.form()
    job_id = int(form.get("job_id"))
    status_raw = form.get("status")

    # Validate job exists
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    # Validate status value
    try:
        status = ApplicationStatus[status_raw]
    except KeyError:
        raise HTTPException(status_code=422, detail=f"Invalid status: {status_raw}")

    # Write the log entry
    log_entry = ApplicationLog(
        job_id=job_id,
        status=status,
    )
    db.add(log_entry)
    await db.commit()

    # Return the refreshed button fragment so HTMX swaps it in
    # The new active status is passed so Jinja2 highlights the right button.
    # NOTE: must match status.name's casing exactly ("APPLIED", not
    # "applied") — ats_buttons.html compares `current == 'APPLIED'` etc.
    # (Bug fix: this used to be status.name.lower(), which silently never
    # matched and meant the just-logged status never got its highlight
    # class until the page was fully reloaded.)
    current = status.name

    return templates.TemplateResponse(
        "partials/ats_buttons.html",
        {
            "request": request,
            "job": job,
            "current": current,
        }
    )


@router.get("/log/{job_id}", response_model=list[ApplicationLogResponse])
async def get_application_history(
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the full application history for a single job.
    """
    stmt = (
        select(ApplicationLog)
        .where(ApplicationLog.job_id == job_id)
        .order_by(desc(ApplicationLog.created_at))
    )
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/pipeline", response_model=list[ApplicationLogResponse])
async def get_pipeline_summary(
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the most recent log entry per job — the active pipeline view.
    Data source for the Visual Dashboard in Phase 3.
    """
    subq = (
        select(
            ApplicationLog.job_id,
            func.max(ApplicationLog.created_at).label("max_created_at"),
        )
        .group_by(ApplicationLog.job_id)
        .subquery()
    )

    stmt = select(ApplicationLog).join(
        subq,
        (ApplicationLog.job_id == subq.c.job_id)
        & (ApplicationLog.created_at == subq.c.max_created_at),
    )

    result = await db.execute(stmt)
    return result.scalars().all()


@router.delete("/log/{log_id}", status_code=204)
async def delete_log_entry(
    log_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Deletes a specific log entry — for correcting mistakes.
    """
    entry = await db.get(ApplicationLog, log_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Log entry {log_id} not found.")

    await db.delete(entry)
    await db.commit()
