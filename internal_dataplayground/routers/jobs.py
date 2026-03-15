from database import get_db  # Import from your new database.py
from models import Job       # Import from your new models.py
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

router = APIRouter(prefix="/jobs", tags=["Jobs"])
templates = Jinja2Templates(directory="templates")

@router.get("", response_class=HTMLResponse)
async def list_jobs_ui(request: Request, db: AsyncSession = Depends(get_db)):
    stmt = select(Job).order_by(desc(Job.fit_score))
    result = await db.execute(stmt)
    jobs = result.scalars().all()
    return templates.TemplateResponse("jobs.html", {"request": request, "jobs": jobs})

@router.get("/detail/{job_id}")
async def get_job_detail(job_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    
    if not job:
        return HTMLResponse(content="Job not found", status_code=404)
        
    return templates.TemplateResponse(
        "partials/job_detail.html", 
        {"request": request, "job": job}
    )