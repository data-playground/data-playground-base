from database import get_db  # Import from your new database.py
from domains.jobs.models import Job
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from core.templating import templates
from routers._helpers import html_error

router = APIRouter(prefix="/jobs", tags=["Jobs"])

@router.get("", response_class=HTMLResponse)
async def list_jobs_ui(request: Request, db: AsyncSession = Depends(get_db)):
    stmt = select(Job).order_by(desc(Job.fit_score))
    result = await db.execute(stmt)
    jobs = result.scalars().all()
    return templates.TemplateResponse("jobs.html", {
        "request": request, 
        "jobs": jobs,
        "active_module": "jobs",
    })

@router.get("/detail/{job_id}")
async def get_job_detail(job_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).where(Job.ID == job_id))  # capital ID
    job = result.scalar_one_or_none()
    if not job:
        return html_error(request, f"Job {job_id} not found.", status_code=404)
    return templates.TemplateResponse(
        "partials/job_detail.html",
        {"request": request, "job": job}
    )
