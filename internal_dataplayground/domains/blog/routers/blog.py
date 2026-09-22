# routers/blog.py
"""
Blog Ideation Module — Kanban & CRUD Router

Split from this file's original monolith (WO#27) — the four HITL/Airflow-
pipeline endpoints (evidence, trigger, review, finalize) now live in
routers/blog_pipeline.py, which owns the Ghostwriter → Refiner → Editor
pipeline specifically. This file keeps the kanban board, BYOI intake, the
generic status/archive/delete/revert endpoints, the Scout trigger, and the
article reader view.

Endpoints:
  GET   /blog                           → Kanban board
  POST  /blog/ideas                     → BYOI: save raw idea, trigger expander DAG
  GET   /blog/ideas/{id}                → Detail drawer partial (HTMX)
  PATCH /blog/ideas/{id}/status         → Generic status update
  PATCH /blog/ideas/{id}/archive        → Move back to backlog
  DELETE /blog/ideas/{id}               → Permanent delete
  PATCH /blog/ideas/{id}/revert         → Move status backward one stage
  POST  /blog/scout                     → Trigger Scout DAG
  GET   /blog/ideas/{id}/article        → Full article reader view

See routers/blog_pipeline.py for: PATCH /blog/ideas/{id}/evidence,
PATCH /blog/ideas/{id}/trigger, PATCH /blog/ideas/{id}/review,
PATCH /blog/ideas/{id}/finalize.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from database import get_db
from domains.blog.models import (
    BlogIdea, BlogIdeaStatus, BlogProjectType,
)
from domains.code_intel.models import CodeFile, CodeProject
from services.airflow_service import trigger_airflow
from core.templating import templates

log = logging.getLogger(__name__)

router = APIRouter(prefix="/blog", tags=["Blog"])

SCOUT_DAG    = "life_os_blog_scout"
EXPANDER_DAG = "life_os_idea_expander"


# ── Kanban board ───────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def blog_kanban(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(BlogIdea)
        .where(BlogIdea.status != BlogIdeaStatus.ARCHIVED) # Filter here
        .order_by(desc(BlogIdea.updated_at))
    )
    all_ideas = result.scalars().all()

    backlog        = [i for i in all_ideas if i.status.kanban_column == "backlog"]
    in_development = [i for i in all_ideas if i.status.kanban_column == "in_development"]
    in_progress    = [i for i in all_ideas if i.status.kanban_column == "in_progress"]
    done           = [i for i in all_ideas if i.status.kanban_column == "done"]

    return templates.TemplateResponse("blog.html", {
        "request": request,
        "backlog": backlog,
        "in_development": in_development,
        "in_progress": in_progress,
        "done": done,
        "active_module": "blog",
    })


# ── BYOI ───────────────────────────────────────────────────────────────────────

@router.post("/ideas", response_class=HTMLResponse)
async def create_idea(
    request: Request,
    raw_idea_input: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    idea = BlogIdea(
        title_concept=raw_idea_input[:80].strip() or "Untitled",
        project_type=BlogProjectType.NEW_BUILD,
        raw_idea_input=raw_idea_input,
        status=BlogIdeaStatus.IDEA_GENERATED,
    )
    db.add(idea)
    await db.commit()
    await db.refresh(idea)

    try:
        await trigger_airflow(EXPANDER_DAG, conf={"idea_id": idea.id})
    except Exception as exc:
        log.warning("Could not trigger enrichment DAG: %s", exc)

    return templates.TemplateResponse(
        "partials/blog_card.html",
        {"request": request, "idea": idea,
         "toast": "Idea saved. Gemini is enriching it in the background."},
    )


# ── Detail drawer ──────────────────────────────────────────────────────────────

@router.get("/ideas/{idea_id}", response_class=HTMLResponse)
async def idea_detail(
    idea_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    idea = await db.get(BlogIdea, idea_id)
    if not idea:
        return HTMLResponse("Not found", status_code=404)

    files_result = await db.execute(select(CodeFile).order_by(CodeFile.file_name))
    code_files = files_result.scalars().all()

    projects_result = await db.execute(select(CodeProject).order_by(CodeProject.project_name))
    code_projects = projects_result.scalars().all()

    return templates.TemplateResponse(
        "partials/blog_detail.html",
        {"request": request, "idea": idea,
         "code_files": code_files, "code_projects": code_projects},
    )


# ── Generic status update ──────────────────────────────────────────────────────

@router.patch("/ideas/{idea_id}/status", response_class=HTMLResponse)
async def update_status(
    idea_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    form = await request.form()
    try:
        new_status = BlogIdeaStatus(str(form.get("status", "")).strip())
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid status value")

    idea = await db.get(BlogIdea, idea_id)
    if not idea:
        raise HTTPException(status_code=404)

    idea.status     = new_status
    idea.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(idea)

    return templates.TemplateResponse(
        "partials/blog_card.html",
        {"request": request, "idea": idea},
    )


# ── Archive / Delete ───────────────────────────────────────────────────────────

@router.patch("/ideas/{idea_id}/archive", response_class=HTMLResponse)
async def archive_idea(
    idea_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    idea = await db.get(BlogIdea, idea_id)
    if not idea:
        raise HTTPException(status_code=404)

    # 1. Update status to ARCHIVED (ensure this is added to your Enum in models.py)
    idea.status = BlogIdeaStatus.ARCHIVED
    idea.updated_at = datetime.utcnow()
    
    await db.commit()

    # 2. Return an empty response with a Trigger header
    # This avoids the "weird behavior" by not sending back HTML to be swapped incorrectly
    response = templates.TemplateResponse(
        "partials/blog_card.html",
        {"request": request, "idea": idea}
    )
    
    # This trigger tells the frontend to remove the card from the UI
    response.headers["HX-Trigger"] = f'{{"ideaArchived": {idea_id}}}'
    
    return response


@router.delete("/ideas/{idea_id}", response_class=HTMLResponse)
async def delete_idea(
    idea_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    idea = await db.get(BlogIdea, idea_id)
    if not idea:
        raise HTTPException(status_code=404)
    await db.delete(idea)
    await db.commit()
    return HTMLResponse("")
    
@router.patch("/ideas/{idea_id}/revert", response_class=HTMLResponse)
async def revert_idea_status(
    idea_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    idea = await db.get(BlogIdea, idea_id)
    if not idea:
        raise HTTPException(status_code=404)

    curr = idea.status
    
    # Define the "Backwards" logic
    if curr in [BlogIdeaStatus.IDEA_GENERATED, BlogIdeaStatus.WAITING_FOR_WRITING_TRIGGER]:
        idea.status = BlogIdeaStatus.ARCHIVED
    elif curr in [BlogIdeaStatus.IN_DEVELOPMENT, BlogIdeaStatus.WRITING_IN_PROGRESS]:
        idea.status = BlogIdeaStatus.WAITING_FOR_WRITING_TRIGGER
    elif curr in [BlogIdeaStatus.WAITING_FOR_REVIEW, BlogIdeaStatus.REVIEW_COMPLETED]:
        idea.status = BlogIdeaStatus.IN_DEVELOPMENT
    elif curr in [BlogIdeaStatus.READY_TO_PUBLISH, BlogIdeaStatus.PUBLISHED]:
        idea.status = BlogIdeaStatus.WAITING_FOR_REVIEW
    
    await db.commit()

    # Return the updated card so HTMX can move it to the correct column
    return templates.TemplateResponse(
        "partials/blog_card.html",
        {"request": request, "idea": idea}
    )


# ── Scout trigger ──────────────────────────────────────────────────────────────

@router.post("/scout", response_class=HTMLResponse)
async def trigger_scout(request: Request):
    try:
        run_id = await trigger_airflow(SCOUT_DAG)
        return HTMLResponse(
            f'<p class="scout-ok">✓ Scout triggered (run: {run_id}). '
            f'New ideas will appear in ~2 minutes.</p>'
        )
    except Exception as exc:
        return HTMLResponse(
            f'<p class="scout-error">⚠ Airflow unreachable: {exc}</p>',
            status_code=502,
        )


# ── Article reader ─────────────────────────────────────────────────────────────

@router.get("/ideas/{idea_id}/article", response_class=HTMLResponse)
async def view_article(
    idea_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    idea = await db.get(BlogIdea, idea_id)
    if not idea:
        raise HTTPException(status_code=404)
    return templates.TemplateResponse(
        "blog_article.html",
        {"request": request, "idea": idea},
    )
