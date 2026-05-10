# routers/ci_readme.py
"""
Code Intelligence — README Management

Endpoints:
  POST  /code-intel/projects/{id}/generate-readme   → Generate README inline (no Airflow)
  PATCH /code-intel/projects/{id}/readme             → Save manual edits, mark reviewed
  POST  /code-intel/projects/{id}/push-readme        → Push README to GitHub
  POST  /code-intel/projects/{id}/trigger-readme     → Trigger README Writer DAG (Airflow)
  PATCH /code-intel/projects/{id}/folder-readme      → Save folder-scoped README
  GET   /code-intel/projects/{id}/folder-readme      → Retrieve folder README content
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import CodeProject, ReadmeStatus
from services.airflow_service import trigger_airflow
from services.github_service import get_file_sha, push_file_content
from airflow.agents.blog_agents import agent_readme_writer

log = logging.getLogger(__name__)

router = APIRouter(prefix="/code-intel", tags=["Code Intelligence"])
templates = Jinja2Templates(directory="templates")

README_WRITER_DAG = "life_os_readme_writer"


# ── Inline README generation (no Airflow) ─────────────────────────────────────

@router.post("/projects/{project_id}/generate-readme", response_class=HTMLResponse)
async def generate_readme(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(CodeProject, project_id)
    if not project:
        raise HTTPException(status_code=404)

    narrated_files = [f for f in project.files if f.narration]
    if not narrated_files:
        return templates.TemplateResponse(
            "partials/project_agent_panel.html",
            {
                "request": request,
                "project": project,
                "error": "No file narrations found. Narrate at least one file first.",
            },
        )

    file_summaries = [
        {"path": f.github_path, "narration": f.narration}
        for f in narrated_files
    ]

    try:
        readme = agent_readme_writer(
            project_name=project.project_name,
            file_summaries=file_summaries,
            description=project.description or "",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"README Writer failed: {exc}")

    project.readme_md = readme
    project.readme_status = ReadmeStatus.DRAFT
    project.readme_generated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(project)

    return templates.TemplateResponse(
        "partials/project_agent_panel.html",
        {
            "request": request,
            "project": project,
            "toast": "README generated. Review and edit before pushing.",
        },
    )


# ── Save manual README edits ───────────────────────────────────────────────────

@router.patch("/projects/{project_id}/readme", response_class=HTMLResponse)
async def save_readme_edits(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Saves manual edits and marks the README as reviewed."""
    form = await request.form()
    project = await db.get(CodeProject, project_id)
    if not project:
        raise HTTPException(status_code=404)

    project.readme_md = str(form.get("readme_md", "")).strip()
    project.readme_status = ReadmeStatus.REVIEWED
    project.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(project)

    return templates.TemplateResponse(
        "partials/project_agent_panel.html",
        {
            "request": request,
            "project": project,
            "toast": "README saved and marked as reviewed.",
        },
    )


# ── Push README to GitHub ──────────────────────────────────────────────────────

@router.post("/projects/{project_id}/push-readme", response_class=HTMLResponse)
async def push_readme(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(CodeProject, project_id)
    if not project:
        raise HTTPException(status_code=404)

    if project.readme_status not in (ReadmeStatus.REVIEWED, ReadmeStatus.APPROVED):
        return templates.TemplateResponse(
            "partials/project_agent_panel.html",
            {
                "request": request,
                "project": project,
                "error": "Mark the README as reviewed before pushing.",
            },
        )

    base = (project.github_base_path or "").rstrip("/")
    readme_path = f"{base}/README.md" if base else "README.md"

    try:
        existing_sha = await get_file_sha(project.github_repo, readme_path)
    except Exception:
        existing_sha = None

    try:
        new_sha = await push_file_content(
            repo=project.github_repo,
            path=readme_path,
            content=project.readme_md or "",
            sha=existing_sha,
            commit_message=f"docs: update README for {project.project_name} via Life OS",
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"GitHub push failed: {exc}")

    project.readme_status = ReadmeStatus.PUSHED
    project.readme_sha = new_sha
    project.readme_pushed_at = datetime.utcnow()
    await db.commit()
    await db.refresh(project)

    return templates.TemplateResponse(
        "partials/project_agent_panel.html",
        {
            "request": request,
            "project": project,
            "toast": f"README pushed to GitHub at {readme_path} ✓",
        },
    )


# ── Trigger README Writer DAG (Airflow) ────────────────────────────────────────

@router.post("/projects/{project_id}/trigger-readme")
async def trigger_readme_dag(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Triggers the README Writer DAG in Airflow.

    - No folder_path  → full project README queued.
                        Returns project_agent_panel HTML partial with toast.
    - With folder_path → folder-scoped README queued.
                         Returns JSON {"run_id", "status", "folder_path"}
                         so the frontend can poll for completion.
    """
    form = await request.form()
    folder_path = str(form.get("folder_path", "")).strip().rstrip("/")

    conf: dict = {"project_id": project_id}
    if folder_path:
        conf["folder_path"] = folder_path

    project = await db.get(CodeProject, project_id)
    if not project:
        raise HTTPException(status_code=404)

    try:
        run_id = await trigger_airflow(README_WRITER_DAG, conf)
    except Exception as exc:
        if folder_path:
            return JSONResponse({"error": str(exc)}, status_code=502)
        return templates.TemplateResponse(
            "partials/project_agent_panel.html",
            {
                "request": request,
                "project": project,
                "error": f"Airflow unreachable: {exc}",
            },
        )

    if folder_path:
        return JSONResponse({
            "run_id": run_id,
            "status": "queued",
            "folder_path": folder_path,
        })

    return templates.TemplateResponse(
        "partials/project_agent_panel.html",
        {
            "request": request,
            "project": project,
            "toast": f"README queued (run: {run_id[:8]}…). Badges update when done.",
        },
    )


# ── Folder README — save / retrieve ───────────────────────────────────────────

@router.patch("/projects/{project_id}/folder-readme", response_class=JSONResponse)
async def save_folder_readme(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Saves a folder-scoped README to folder_readme_md WITHOUT touching readme_md.
    Called from the Airflow DAG callback OR from 'Save as Project README' in UI.
    """
    data = await request.json()
    content = data.get("content", "").strip()
    folder_path = data.get("folder_path", "").strip()
    save_as_project = bool(data.get("save_as_project", False))

    project = await db.get(CodeProject, project_id)
    if not project:
        raise HTTPException(status_code=404)

    project.folder_readme_md = content
    project.folder_readme_path = folder_path
    project.folder_readme_generated_at = datetime.utcnow()

    if save_as_project:
        project.readme_md = content
        project.readme_status = ReadmeStatus.REVIEWED
        project.readme_generated_at = datetime.utcnow()

    await db.commit()
    return {"ok": True, "saved_as_project": save_as_project}


@router.get("/projects/{project_id}/folder-readme")
async def get_folder_readme(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Returns the stored folder README content and which folder it covers."""
    project = await db.get(CodeProject, project_id)
    if not project or not project.folder_readme_md:
        raise HTTPException(status_code=404, detail="No folder README available")

    return {
        "content": project.folder_readme_md,
        "folder_path": project.folder_readme_path or "",
        "generated_at": (
            str(project.folder_readme_generated_at)
            if project.folder_readme_generated_at else None
        ),
    }
