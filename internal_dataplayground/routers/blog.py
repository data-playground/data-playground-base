# routers/blog.py
"""
Blog Ideation Module

Endpoints:
  GET  /blog                        → Kanban board
  POST /blog/ideas                  → BYOI — submit raw idea, Gemini expands to blueprint
  GET  /blog/ideas/{id}             → Detail drawer partial (HTMX)
  PATCH /blog/ideas/{id}/evidence   → Save code + author notes → advances to waiting_for_writing_trigger
  PATCH /blog/ideas/{id}/trigger    → Call Airflow REST API → starts Creator DAG
  PATCH /blog/ideas/{id}/review     → Save review notes → advances to review_completed
  PATCH /blog/ideas/{id}/status     → Generic status update
  POST /blog/scout                  → Call Airflow REST API → triggers Scout DAG
  GET  /blog/ideas/{id}/article     → Full article view
"""

import json
import logging
import httpx
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from google import genai

from database import get_db, get_key
from models import BlogIdea, BlogIdeaStatus, BlogProjectType

log = logging.getLogger(__name__)

router = APIRouter(prefix="/blog", tags=["Blog"])
templates = Jinja2Templates(directory="templates")

# ── Airflow config ─────────────────────────────────────────────────────────────
# Airflow is running on the host machine, reachable from Docker via host.docker.internal
AIRFLOW_BASE    = "http://airflow-webserver:8080/api/v1"
AIRFLOW_USER    = "admin"
AIRFLOW_PASS    = "admin"          # swap to get_key("AirflowPassword") if needed
SCOUT_DAG_ID    = "life_os_blog_scout"
CREATOR_DAG_ID  = "life_os_blog_creator"


def _airflow_headers() -> dict:
    import base64
    token = base64.b64encode(f"{AIRFLOW_USER}:{AIRFLOW_PASS}".encode()).decode()
    return {"Authorization": f"Basic {token}", "Content-Type": "application/json"}


async def _trigger_airflow_dag(dag_id: str, conf: dict = {}) -> str:
    """Triggers an Airflow DAG and returns the run_id."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{AIRFLOW_BASE}/dags/{dag_id}/dagRuns",
            headers=_airflow_headers(),
            json={"conf": conf},
        )
        resp.raise_for_status()
        return resp.json().get("dag_run_id", "")


# ── Gemini blueprint expander ──────────────────────────────────────────────────

def _expand_idea_to_blueprint(raw_input: str) -> dict:
    """
    Takes raw user notes/code and returns a structured blueprint dict:
    {title_concept, project_type, the_build, the_narrative, the_selling_point}
    """
    client = genai.Client(api_key=get_key("Gemini-API"))

    prompt = f"""You are a technical content strategist for a data engineer's portfolio blog.
A user has submitted raw notes or code. Expand it into a structured article blueprint.

Raw input:
\"\"\"
{raw_input}
\"\"\"

Respond ONLY with a JSON object with exactly these keys:
{{
  "title_concept": "A compelling, SEO-friendly article title (max 80 chars)",
  "project_type": "existing_asset" or "new_build",
  "the_build": "2-3 sentences describing what was technically built and the stack used",
  "the_narrative": "2-3 sentences on the story arc — what problem this solves and why it matters",
  "the_selling_point": "1 sentence on the unique insight or skill this demonstrates to employers/readers"
}}

No markdown, no explanation. Raw JSON only."""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt,
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        log.warning("Blueprint expansion failed: %s", exc)
        return {
            "title_concept": "Untitled — edit me",
            "project_type": "new_build",
            "the_build": "",
            "the_narrative": raw_input[:500],
            "the_selling_point": "",
        }


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


# ── BYOI — create new idea ─────────────────────────────────────────────────────

@router.post("/ideas", response_class=HTMLResponse)
async def create_idea(
    request: Request,
    raw_idea_input: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    blueprint = _expand_idea_to_blueprint(raw_idea_input)

    try:
        ptype = BlogProjectType(blueprint.get("project_type", "new_build"))
    except ValueError:
        ptype = BlogProjectType.NEW_BUILD

    idea = BlogIdea(
        title_concept    = blueprint.get("title_concept", "Untitled"),
        project_type     = ptype,
        the_build        = blueprint.get("the_build"),
        the_narrative    = blueprint.get("the_narrative"),
        the_selling_point= blueprint.get("the_selling_point"),
        raw_idea_input   = raw_idea_input,
        status           = BlogIdeaStatus.IDEA_GENERATED,
    )
    db.add(idea)
    await db.commit()
    await db.refresh(idea)

    # Return a new card partial to prepend into the backlog column
    return templates.TemplateResponse(
        "partials/blog_card.html",
        {"request": request, "idea": idea},
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
    return templates.TemplateResponse(
        "partials/blog_detail.html",
        {"request": request, "idea": idea},
    )


# ── HITL Checkpoint 1 — save evidence ─────────────────────────────────────────

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

    idea.code_content = str(form.get("code_content", "")).strip() or None
    idea.author_notes = str(form.get("author_notes", "")).strip() or None
    idea.status       = BlogIdeaStatus.WAITING_FOR_WRITING_TRIGGER
    idea.updated_at   = datetime.utcnow()
    await db.commit()
    await db.refresh(idea)

    return templates.TemplateResponse(
        "partials/blog_detail.html",
        {"request": request, "idea": idea,
         "toast": "Evidence saved. Ready to trigger the Creator pipeline."},
    )


# ── Trigger Creator DAG ────────────────────────────────────────────────────────

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
        raise HTTPException(status_code=409, detail="Idea is already in the pipeline.")

    try:
        run_id = await _trigger_airflow_dag(
            CREATOR_DAG_ID,
            conf={"idea_id": idea_id},
        )
        idea.airflow_run_id = run_id
        idea.status         = BlogIdeaStatus.WRITING_IN_PROGRESS
    except Exception as exc:
        log.warning("Airflow trigger failed: %s", exc)
        # Soft fail — mark status but surface the error to the user
        return templates.TemplateResponse(
            "partials/blog_detail.html",
            {"request": request, "idea": idea,
             "error": f"Airflow unreachable: {exc}. Is Airflow running on port 8080?"},
        )

    idea.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(idea)

    return templates.TemplateResponse(
        "partials/blog_detail.html",
        {"request": request, "idea": idea,
         "toast": "Creator DAG triggered. Check Airflow for progress."},
    )


# ── HITL Checkpoint 2 — review draft ──────────────────────────────────────────

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
    idea.status            = BlogIdeaStatus.REVIEW_COMPLETED
    idea.updated_at        = datetime.utcnow()
    await db.commit()
    await db.refresh(idea)

    return templates.TemplateResponse(
        "partials/blog_detail.html",
        {"request": request, "idea": idea,
         "toast": "Review saved. The Refiner + SEO agents will now finalize the article."},
    )


# ── Generic status update ──────────────────────────────────────────────────────

@router.patch("/ideas/{idea_id}/status", response_class=HTMLResponse)
async def update_status(
    idea_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    form = await request.form()
    status_str = str(form.get("status", "")).strip()
    try:
        new_status = BlogIdeaStatus(status_str)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid status: {status_str}")

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


# ── Trigger Scout DAG ──────────────────────────────────────────────────────────

@router.post("/scout", response_class=HTMLResponse)
async def trigger_scout(request: Request):
    try:
        run_id = await _trigger_airflow_dag(SCOUT_DAG_ID)
        return HTMLResponse(
            f'<p class="scout-ok">✓ Scout DAG triggered (run: {run_id}). '
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
