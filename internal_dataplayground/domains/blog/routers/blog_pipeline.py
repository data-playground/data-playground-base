# routers/blog_pipeline.py
"""
Blog Ideation Module — HITL / Pipeline Router

Split out of routers/blog.py (WO#27). Owns the four endpoints that drive
the Ghostwriter → Refiner → Editor pipeline and the human-in-the-loop
steps between each trigger. The kanban/CRUD surface (board, BYOI intake,
status/archive/delete/revert, Scout trigger, article reader) stays in
routers/blog.py — see that file's docstring for the full endpoint split.

Endpoints:
  PATCH /blog/ideas/{id}/evidence       → Save code + author notes + difficulty (HITL 1)
  PATCH /blog/ideas/{id}/trigger        → Trigger Ghostwriter DAG
  PATCH /blog/ideas/{id}/review         → Save review notes (HITL 2)
  PATCH /blog/ideas/{id}/finalize       → Trigger Refiner+Editor DAG
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db
from domains.blog.models import BlogIdea, BlogIdeaStatus, DIFFICULTY_LEVELS
from domains.code_intel.models import CodeFile, CodeProject
from services.airflow_service import trigger_airflow
from core.templating import templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/blog", tags=["Blog"])

CREATOR_DAG   = "life_os_blog_creator"
FINALIZER_DAG = "life_os_blog_finalizer"


# ── HITL 1 — Save evidence ─────────────────────────────────────────────────────

@router.patch("/ideas/{idea_id}/evidence", response_class=HTMLResponse)
async def save_evidence(
    idea_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    form = await request.form()
    idea = await db.get(BlogIdea, idea_id)
    if not idea:
        raise HTTPException(status_code=404)

    idea.code_content  = str(form.get("code_content", "")).strip() or None
    idea.author_notes  = str(form.get("author_notes", "")).strip() or None

    difficulty_raw = str(form.get("difficulty", "")).strip()
    if difficulty_raw in DIFFICULTY_LEVELS:
        idea.difficulty = difficulty_raw

    code_file_id    = form.get("code_file_id")
    code_project_id = form.get("code_project_id")
    idea.code_file_id    = int(code_file_id)    if code_file_id    else None
    idea.code_project_id = int(code_project_id) if code_project_id else None

    idea.status     = BlogIdeaStatus.WAITING_FOR_WRITING_TRIGGER
    idea.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(idea)

    files_result = await db.execute(select(CodeFile).order_by(CodeFile.file_name))
    projects_result = await db.execute(select(CodeProject).order_by(CodeProject.project_name))

    return templates.TemplateResponse(
        "partials/blog_detail.html",
        {"request": request, "idea": idea,
         "code_files": files_result.scalars().all(),
         "code_projects": projects_result.scalars().all(),
         "toast": "Evidence saved. Ready to trigger the Ghostwriter."},
    )


# ── Trigger 1 — Ghostwriter DAG ───────────────────────────────────────────────

@router.patch("/ideas/{idea_id}/trigger", response_class=HTMLResponse)
async def trigger_creator(
    idea_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    idea = await db.get(BlogIdea, idea_id)
    if not idea:
        raise HTTPException(status_code=404)

    # IN_DEVELOPMENT is allowed — triggering auto-advances status
    if idea.status not in (
        BlogIdeaStatus.IDEA_GENERATED,
        BlogIdeaStatus.WAITING_FOR_WRITING_TRIGGER,
        BlogIdeaStatus.IN_DEVELOPMENT,
    ):
        return templates.TemplateResponse(
            "partials/blog_detail.html",
            {"request": request, "idea": idea,
             "error": f"Cannot trigger from status: {idea.status.label}"},
        )

    try:
        run_id = await trigger_airflow(CREATOR_DAG, conf={"idea_id": idea_id})
        idea.airflow_run_id = run_id
        idea.status = BlogIdeaStatus.WRITING_IN_PROGRESS
    except Exception as exc:
        log.warning("Airflow trigger failed: %s", exc)
        return templates.TemplateResponse(
            "partials/blog_detail.html",
            {"request": request, "idea": idea,
             "error": f"Airflow unreachable: {exc}"},
        )

    idea.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(idea)

    return templates.TemplateResponse(
        "partials/blog_detail.html",
        {"request": request, "idea": idea,
         "toast": f"Ghostwriter DAG triggered (run: {run_id}). Check Airflow for progress."},
    )


# ── HITL 2 — Save review notes ────────────────────────────────────────────────

@router.patch("/ideas/{idea_id}/review", response_class=HTMLResponse)
async def save_review(
    idea_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    form = await request.form()
    idea = await db.get(BlogIdea, idea_id)
    if not idea:
        raise HTTPException(status_code=404)

    idea.user_review_notes = str(form.get("user_review_notes", "")).strip() or None
    idea.status            = BlogIdeaStatus.WAITING_FOR_REVIEW
    idea.updated_at        = datetime.utcnow()
    await db.commit()
    await db.refresh(idea)

    return templates.TemplateResponse(
        "partials/blog_detail.html",
        {"request": request, "idea": idea,
         "toast": "Review notes saved. Click Finalize when ready."},
    )


# ── Trigger 2 — Finalizer DAG ─────────────────────────────────────────────────

@router.patch("/ideas/{idea_id}/finalize", response_class=HTMLResponse)
async def trigger_finalizer(
    idea_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    idea = await db.get(BlogIdea, idea_id)
    if not idea:
        raise HTTPException(status_code=404)

    if not idea.draft_v1:
        return templates.TemplateResponse(
            "partials/blog_detail.html",
            {"request": request, "idea": idea,
             "error": "No draft found. Run the Ghostwriter first."},
        )

    try:
        run_id = await trigger_airflow(FINALIZER_DAG, conf={"idea_id": idea_id})
        idea.airflow_run_id = run_id
        idea.status = BlogIdeaStatus.REVIEW_COMPLETED
    except Exception as exc:
        log.warning("Finalizer DAG trigger failed: %s", exc)
        return templates.TemplateResponse(
            "partials/blog_detail.html",
            {"request": request, "idea": idea,
             "error": f"Airflow unreachable: {exc}"},
        )

    idea.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(idea)

    return templates.TemplateResponse(
        "partials/blog_detail.html",
        {"request": request, "idea": idea,
         "toast": f"Refiner + Editor triggered (run: {run_id})."},
    )
