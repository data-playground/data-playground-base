# domains/jobs/routers/job_config.py
"""
Job Scout configuration — search keywords + ATS company watchlist.

Endpoints:
  GET    /jobs/config                      -> settings page
  POST   /jobs/config/keywords              -> add a search keyword
  PATCH  /jobs/config/keywords/{id}/toggle   -> toggle active
  DELETE /jobs/config/keywords/{id}          -> remove
  GET    /jobs/config/candidates             -> companies worth watching, not yet followed
  POST   /jobs/config/watched                -> add a company to the ATS watchlist
  PATCH  /jobs/config/watched/{id}             -> update greenhouse/lever slugs or toggle active
  DELETE /jobs/config/watched/{id}             -> remove from watchlist

life_os_job_scout.py reads job_search_keywords (is_active=True) directly via
dag_db raw SQL — no code change needed there when you add/remove a keyword
here. A future ATS DAG would read watched_companies the same way.
"""
import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.jobs.models import JobSearchKeyword, WatchedCompany, Job, JobScoutRunLog
from routers._helpers import html_error

from core.templating import templates

router = APIRouter(prefix="/jobs/config", tags=["Job Config"])

# Minimum signal required before a company is surfaced as a watchlist candidate.
CANDIDATE_MIN_POSTINGS = 2
CANDIDATE_MIN_AVG_FIT = 80


async def _get_health_summary(db: AsyncSession) -> list[dict]:
    """
    Latest run per DAG from job_scout_run_log, written by life_os_job_scout.py,
    life_os_job_scout_ats.py (and read again by life_os_daily_digest.py in
    Airflow via job_scout_health.py — same table, two different read paths,
    since the FastAPI container can't share Airflow's dag_db/pymysql setup).
    """
    latest_subq = (
        select(
            JobScoutRunLog.dag_id,
            func.max(JobScoutRunLog.run_at).label("max_run_at"),
        )
        .group_by(JobScoutRunLog.dag_id)
        .subquery()
    )
    result = await db.execute(
        select(JobScoutRunLog).join(
            latest_subq,
            (JobScoutRunLog.dag_id == latest_subq.c.dag_id)
            & (JobScoutRunLog.run_at == latest_subq.c.max_run_at),
        )
    )
    return [
        {
            "dag_id": r.dag_id, "run_at": r.run_at, "items_found": r.items_found,
            "new_items": r.new_items, "items_loaded": r.items_loaded,
            "status": r.status, "message": r.message,
        }
        for r in result.scalars().all()
    ]


@router.get("", response_class=HTMLResponse)
async def config_page(request: Request, db: AsyncSession = Depends(get_db)):
    keywords_result = await db.execute(select(JobSearchKeyword).order_by(JobSearchKeyword.keyword))
    watched_result = await db.execute(select(WatchedCompany).order_by(WatchedCompany.company_name))
    candidates = await _get_candidate_companies(db)
    health = await _get_health_summary(db)

    return templates.TemplateResponse("job_config.html", {
        "request": request,
        "active_module": "jobs_settings",
        "keywords": keywords_result.scalars().all(),
        "watched": watched_result.scalars().all(),
        "candidates": candidates,
        "health": health,
    })


# ── SEARCH KEYWORDS ──────────────────────────────────────────────────────────

@router.post("/keywords", response_class=HTMLResponse)
async def add_keyword(request: Request, db: AsyncSession = Depends(get_db)):
    form = await request.form()
    keyword = str(form.get("keyword", "")).strip()
    if not keyword:
        return html_error(request, "Enter a search keyword first.", status_code=422)

    existing = await db.execute(select(JobSearchKeyword).where(JobSearchKeyword.keyword == keyword))
    if existing.scalar_one_or_none():
        return html_error(request, "That keyword is already in the list.", status_code=409)

    db.add(JobSearchKeyword(keyword=keyword))
    await db.commit()

    result = await db.execute(select(JobSearchKeyword).order_by(JobSearchKeyword.keyword))
    return templates.TemplateResponse("partials/job_keyword_list.html", {
        "request": request, "keywords": result.scalars().all(),
    })


@router.patch("/keywords/{keyword_id}/toggle", response_class=HTMLResponse)
async def toggle_keyword(keyword_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    kw = await db.get(JobSearchKeyword, keyword_id)
    if not kw:
        return html_error(request, "Keyword not found.", status_code=404)
    kw.is_active = not kw.is_active
    kw.updated_at = datetime.datetime.utcnow()
    await db.commit()

    result = await db.execute(select(JobSearchKeyword).order_by(JobSearchKeyword.keyword))
    return templates.TemplateResponse("partials/job_keyword_list.html", {
        "request": request, "keywords": result.scalars().all(),
    })


@router.delete("/keywords/{keyword_id}", response_class=HTMLResponse)
async def delete_keyword(keyword_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    kw = await db.get(JobSearchKeyword, keyword_id)
    if kw:
        await db.delete(kw)
        await db.commit()

    result = await db.execute(select(JobSearchKeyword).order_by(JobSearchKeyword.keyword))
    return templates.TemplateResponse("partials/job_keyword_list.html", {
        "request": request, "keywords": result.scalars().all(),
    })


# ── ATS WATCHLIST + CANDIDATE PROMOTION SIGNAL ────────────────────────────────

async def _get_candidate_companies(db: AsyncSession) -> list[dict]:
    """
    Surfaces companies from the LinkedIn scrape that keep showing up with
    strong fit scores but aren't on the ATS watchlist yet — the "should I
    add this company's Greenhouse/Lever board?" signal.
    """
    watched_result = await db.execute(select(WatchedCompany.company_name))
    already_watched = {row[0].lower() for row in watched_result.all()}

    stats_result = await db.execute(
        select(
            Job.company_name,
            func.count(Job.ID).label("posting_count"),
            func.avg(Job.fit_score).label("avg_fit_score"),
            func.max(Job.fit_score).label("max_fit_score"),
        )
        .where(Job.company_name.is_not(None))
        .group_by(Job.company_name)
        .having(func.count(Job.ID) >= CANDIDATE_MIN_POSTINGS)
        .having(func.avg(Job.fit_score) >= CANDIDATE_MIN_AVG_FIT)
        .order_by(desc("avg_fit_score"))
        .limit(15)
    )

    return [
        {
            "company_name":  row.company_name,
            "posting_count": row.posting_count,
            "avg_fit_score": round(row.avg_fit_score or 0),
            "max_fit_score": row.max_fit_score,
        }
        for row in stats_result.all()
        if row.company_name.lower() not in already_watched
    ]


@router.get("/candidates", response_class=HTMLResponse)
async def candidate_companies(request: Request, db: AsyncSession = Depends(get_db)):
    candidates = await _get_candidate_companies(db)
    return templates.TemplateResponse("partials/job_candidate_list.html", {
        "request": request, "candidates": candidates,
    })


@router.post("/watched", response_class=HTMLResponse)
async def add_watched_company(request: Request, db: AsyncSession = Depends(get_db)):
    form = await request.form()
    company_name = str(form.get("company_name", "")).strip()
    if not company_name:
        return html_error(request, "Company name is required.", status_code=422)

    existing = await db.execute(select(WatchedCompany).where(WatchedCompany.company_name == company_name))
    if existing.scalar_one_or_none():
        return html_error(request, f"{company_name} is already on the watchlist.", status_code=409)

    db.add(WatchedCompany(
        company_name=company_name,
        greenhouse_slug=str(form.get("greenhouse_slug", "")).strip() or None,
        lever_slug=str(form.get("lever_slug", "")).strip() or None,
        source_note=str(form.get("source_note", "")).strip() or None,
    ))
    await db.commit()

    return await _render_watched_panel(request, db)


@router.patch("/watched/{company_id}", response_class=HTMLResponse)
async def update_watched_company(company_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    form = await request.form()
    company = await db.get(WatchedCompany, company_id)
    if not company:
        return html_error(request, "Company not found.", status_code=404)

    if "greenhouse_slug" in form:
        company.greenhouse_slug = str(form.get("greenhouse_slug", "")).strip() or None
    if "lever_slug" in form:
        company.lever_slug = str(form.get("lever_slug", "")).strip() or None
    if "toggle_active" in form:
        company.is_active = not company.is_active
    company.updated_at = datetime.datetime.utcnow()
    await db.commit()

    return await _render_watched_panel(request, db)


@router.delete("/watched/{company_id}", response_class=HTMLResponse)
async def delete_watched_company(company_id: int, request: Request, db: AsyncSession = Depends(get_db)):
    company = await db.get(WatchedCompany, company_id)
    if company:
        await db.delete(company)
        await db.commit()

    return await _render_watched_panel(request, db)


async def _render_watched_panel(request: Request, db: AsyncSession) -> HTMLResponse:
    """Shared re-render for all watchlist mutations — keeps the watched table
    and the candidates list in sync (a newly-added company should immediately
    drop out of the candidates panel)."""
    watched_result = await db.execute(select(WatchedCompany).order_by(WatchedCompany.company_name))
    candidates = await _get_candidate_companies(db)
    return templates.TemplateResponse("partials/job_watched_panel.html", {
        "request": request,
        "watched": watched_result.scalars().all(),
        "candidates": candidates,
    })


# ── SLUG AUTO-DETECTION ───────────────────────────────────────────────────────

@router.post("/watched/detect", response_class=HTMLResponse)
async def detect_slugs(request: Request):
    """
    Best-effort guess at a company's Greenhouse/Lever slug from its name —
    saves the "open their careers page, find the URL, copy the token" step
    for the common case where the slug is just the company name. Always
    shown as a suggestion to confirm, never auto-saved — a generic guess can
    occasionally land on an unrelated company with a similar name.
    """
    from services.ats_slug_service import guess_ats_slugs

    form = await request.form()
    company_name = str(form.get("company_name", "")).strip()
    if not company_name:
        return HTMLResponse(
            '<span style="font-size:10px;color:var(--red);">Enter a company name first.</span>'
        )

    guesses = await guess_ats_slugs(company_name)
    return templates.TemplateResponse("partials/job_slug_guess.html", {
        "request": request, "guesses": guesses,
    })
