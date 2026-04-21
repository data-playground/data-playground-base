# routers/blog.py
"""
Blog Ideation Module — Full Pipeline Router

Endpoints:
  GET  /blog                           → Kanban board (backlog / writing / done)
  POST /blog/ideas                     → BYOI: expand raw idea → blueprint → DB
  GET  /blog/ideas/{id}                → Detail drawer partial (HTMX)
  PATCH /blog/ideas/{id}/evidence      → Save code + author notes (HITL 1)
  PATCH /blog/ideas/{id}/trigger       → Trigger Ghostwriter DAG (life_os_blog_creator)
  PATCH /blog/ideas/{id}/review        → Save review notes (HITL 2)
  PATCH /blog/ideas/{id}/finalize      → Trigger Refiner+Editor DAG (life_os_blog_finalizer)
  PATCH /blog/ideas/{id}/status        → Generic status update
  POST  /blog/scout                    → Trigger Scout DAG (life_os_blog_scout)
  GET   /blog/ideas/{id}/article       → Full article view
"""

import json
import logging
import httpx
import base64
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from database import get_db, get_key
from models import BlogIdea, BlogIdeaStatus, BlogProjectType, CodeFile, CodeProject

log = logging.getLogger(__name__)

router = APIRouter(prefix="/blog", tags=["Blog"])
templates = Jinja2Templates(directory="templates")

# ── Airflow config ─────────────────────────────────────────────────────────────
AIRFLOW_BASE   = "http://airflow-webserver:8080/api/v1"
AIRFLOW_USER   = "admin"
SCOUT_DAG      = "life_os_blog_scout"
CREATOR_DAG    = "life_os_blog_creator"
FINALIZER_DAG  = "life_os_blog_finalizer"


def _airflow_headers() -> dict:
    # Password pulled from GCP Secret Manager — same as DB password in your setup
    password = get_key("Airflow-Admin-Password")  # or reuse DB_PASSWORD if same
    token = base64.b64encode(f"{AIRFLOW_USER}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


async def _trigger_dag(dag_id: str, conf: dict = {}) -> str:
    """Triggers an Airflow DAG and returns the run_id."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{AIRFLOW_BASE}/dags/{dag_id}/dagRuns",
            headers=_airflow_headers(),
            json={"conf": conf},
        )
        resp.raise_for_status()
        return resp.json().get("dag_run_id", "unknown")


# ── Gemini blueprint expander (for BYOI) ───────────────────────────────────────

# def _expand_idea_to_blueprint(raw_input: str) -> dict:
    # from agents.blog_agents import agent_idea_expander
    # return agent_idea_expander(raw_input)
    
def _expand_idea_to_blueprint(user_idea: str) -> dict:
    import sys
    import os
    
    # Ensure the airflow agents folder is reachable from FastAPI's working dir
    airflow_path = os.path.join(os.path.dirname(__file__), '..', 'airflow')
    airflow_path = os.path.abspath(airflow_path)
    if airflow_path not in sys.path:
        sys.path.insert(0, airflow_path)
    
    from agents.blog_agents import agent_idea_expander
    return agent_idea_expander(user_idea)

# ── Kanban board ───────────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def blog_kanban(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(BlogIdea).order_by(desc(BlogIdea.updated_at))
    )
    all_ideas = result.scalars().all()

    backlog     = [i for i in all_ideas if i.status.kanban_column == "backlog"]
    in_progress = [i for i in all_ideas if i.status.kanban_column == "in_progress"]
    done        = [i for i in all_ideas if i.status.kanban_column == "done"]

    return templates.TemplateResponse("blog.html", {
        "request": request,
        "backlog": backlog,
        "in_progress": in_progress,
        "done": done,
        "active_module": "blog",
    })


# ── BYOI — create new idea from raw input ──────────────────────────────────────

@router.post("/ideas", response_class=HTMLResponse)
async def create_idea(
    request: Request,
    raw_idea_input: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Receives raw user notes from the BYOI modal.
    Calls Idea Expander agent, stores structured blueprint in blog_ideas.
    Returns a new card partial for HTMX to prepend into the backlog column.
    """
    # Save immediately — no Gemini call here
    idea = BlogIdea(
        title_concept=raw_idea_input[:80].strip() or "Untitled",
        project_type=BlogProjectType.NEW_BUILD,
        raw_idea_input=raw_idea_input,
        status=BlogIdeaStatus.IDEA_GENERATED,
    )
    db.add(idea)
    await db.commit()
    await db.refresh(idea)

    # Trigger enrichment DAG in background — fire and forget
    try:
        await _trigger_dag("life_os_idea_expander", conf={"idea_id": idea.id})
    except Exception as exc:
        log.warning("Could not trigger enrichment DAG: %s", exc)
        # Non-fatal — idea is saved, enrichment can be retried manually

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
    
    # Fetch available code files and projects for the dropdown
    files_result = await db.execute(
        select(CodeFile).order_by(CodeFile.file_name)
    )
    code_files = files_result.scalars().all()
    
    projects_result = await db.execute(
        select(CodeProject).order_by(CodeProject.project_name)
    )
    code_projects = projects_result.scalars().all()
    
    return templates.TemplateResponse(
        "partials/blog_detail.html",
        {
            "request": request,
            "idea": idea,
            "code_files": code_files,
            "code_projects": code_projects,
        },
    )

# ── HITL 1 — Save evidence (code snippets + author notes) ─────────────────────

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
    
    # Save the code file / project link
    code_file_id = form.get("code_file_id")
    code_project_id = form.get("code_project_id")
    idea.code_file_id    = int(code_file_id) if code_file_id else None
    idea.code_project_id = int(code_project_id) if code_project_id else None
    
    idea.status     = BlogIdeaStatus.WAITING_FOR_WRITING_TRIGGER
    idea.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(idea)

    return templates.TemplateResponse(
        "partials/blog_detail.html",
        {"request": request, "idea": idea,
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

    if idea.status not in (
        BlogIdeaStatus.IDEA_GENERATED,
        BlogIdeaStatus.WAITING_FOR_WRITING_TRIGGER,
    ):
        return templates.TemplateResponse(
            "partials/blog_detail.html",
            {"request": request, "idea": idea,
             "error": f"Cannot trigger from status: {idea.status.label}"},
        )

    try:
        run_id = await _trigger_dag(CREATOR_DAG, conf={"idea_id": idea_id})
        idea.airflow_run_id = run_id
        idea.status = BlogIdeaStatus.WRITING_IN_PROGRESS
    except Exception as exc:
        log.warning("Airflow trigger failed: %s", exc)
        return templates.TemplateResponse(
            "partials/blog_detail.html",
            {"request": request, "idea": idea,
             "error": f"Airflow unreachable: {exc}. Is the airflow-webserver container running?"},
        )

    idea.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(idea)

    return templates.TemplateResponse(
        "partials/blog_detail.html",
        {"request": request, "idea": idea,
         "toast": f"Ghostwriter DAG triggered (run: {run_id}). Check Airflow for progress."},
    )


# ── HITL 2 — Save review notes after reading draft_v1 ─────────────────────────

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
         "toast": "Review notes saved. Click Finalize when ready to run Refiner + Editor."},
    )


# ── Trigger 2 — Finalizer DAG (Refiner + Editor) ──────────────────────────────

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
        run_id = await _trigger_dag(FINALIZER_DAG, conf={"idea_id": idea_id})
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
         "toast": f"Refiner + Editor triggered (run: {run_id}). Will be READY_TO_PUBLISH shortly."},
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


# ── Scout DAG trigger ──────────────────────────────────────────────────────────

@router.post("/scout", response_class=HTMLResponse)
async def trigger_scout(request: Request):
    try:
        run_id = await _trigger_dag(SCOUT_DAG)
        return HTMLResponse(
            f'<p class="scout-ok">✓ Scout triggered (run: {run_id}). '
            f'New ideas will appear in ~2 minutes.</p>'
        )
    except Exception as exc:
        return HTMLResponse(
            f'<p class="scout-error">⚠ Airflow unreachable: {exc}</p>',
            status_code=502,
        )


# ── Full article view ──────────────────────────────────────────────────────────

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
    # Return empty string — HTMX will remove the card
    return HTMLResponse("")


@router.patch("/ideas/{idea_id}/archive", response_class=HTMLResponse)
async def archive_idea(
    idea_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Move any idea back to idea_generated (backlog)."""
    idea = await db.get(BlogIdea, idea_id)
    if not idea:
        raise HTTPException(status_code=404)
    idea.status = BlogIdeaStatus.IDEA_GENERATED
    idea.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(idea)
    return templates.TemplateResponse(
        "partials/blog_detail.html",
        {"request": request, "idea": idea, "toast": "Moved back to backlog."},
    )