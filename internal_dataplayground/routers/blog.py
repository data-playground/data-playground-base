# routers/blog.py
"""
Blog Ideation Module — Full Pipeline Router

Endpoints:
  GET   /blog                           → Kanban board
  POST  /blog/ideas                     → BYOI: save raw idea, trigger expander DAG
  GET   /blog/ideas/{id}                → Detail drawer partial (HTMX)
  PATCH /blog/ideas/{id}/evidence       → Save code + author notes + difficulty (HITL 1)
  PATCH /blog/ideas/{id}/trigger        → Trigger Ghostwriter DAG
  PATCH /blog/ideas/{id}/review         → Save review notes (HITL 2)
  PATCH /blog/ideas/{id}/finalize       → Trigger Refiner+Editor DAG
  PATCH /blog/ideas/{id}/status         → Generic status update
  PATCH /blog/ideas/{id}/archive        → Move back to backlog
  DELETE /blog/ideas/{id}               → Permanent delete
  POST  /blog/scout                     → Trigger Scout DAG
  GET   /blog/ideas/{id}/article        → Full article reader view
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from database import get_db
from models import (
    BlogIdea, BlogIdeaStatus, BlogProjectType,
    CodeFile, CodeProject, DIFFICULTY_LEVELS,
)
from services.airflow_service import trigger_airflow

log = logging.getLogger(__name__)

router = APIRouter(prefix="/blog", tags=["Blog"])
templates = Jinja2Templates(directory="templates")

SCOUT_DAG     = "life_os_blog_scout"
CREATOR_DAG   = "life_os_blog_creator"
FINALIZER_DAG = "life_os_blog_finalizer"
EXPANDER_DAG  = "life_os_idea_expander"


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
