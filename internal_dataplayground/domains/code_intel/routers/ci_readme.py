# routers/ci_readme.py
"""
Code Intelligence — Project README Management

Endpoints:
  POST  /code-intel/projects/{id}/generate-readme   → Generate README inline (no Airflow)
  PATCH /code-intel/projects/{id}/readme             → Save manual edits, mark reviewed
  POST  /code-intel/projects/{id}/push-readme        → Push README to GitHub
  POST  /code-intel/projects/{id}/trigger-readme     → Trigger README Writer DAG (Airflow)

Folder-scoped README storage (PATCH/GET .../folder-readme) lives in
ci_folder_readme.py (WO#24 Part A) — this file imports its
``_get_latest_folder_readme`` helper to keep the folder-README panel
populated on the responses below.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.code_intel.models import CodeProject, ReadmeStatus
from services.airflow_service import trigger_airflow
from services.github_service import get_file_sha, push_file_content
from airflow.agents.blog_agents import agent_readme_writer
from core.templating import templates
from domains.code_intel.routers.ci_folder_readme import _get_latest_folder_readme

log = logging.getLogger(__name__)

router = APIRouter(prefix="/code-intel", tags=["Code Intelligence"])

README_WRITER_DAG = "life_os_readme_writer"


# ── Inline README generation (no Airflow) ─────────────────────────────────────

@router.post("/projects/{project_id}/generate-readme", response_class=HTMLResponse)
async def generate_readme(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Generate a full project README inline (synchronously, no Airflow).

    Synthesizes narrations from every narrated CodeFile in the project into
    a single README draft via the README Writer agent.

    Args:
        project_id: CodeProject to generate a README for.
        request: Current request, forwarded to the template response.
        db: Async DB session (injected).

    Returns:
        The re-rendered ``partials/project_agent_panel.html`` fragment,
        either with an error (no narrated files yet) or the new draft.

    Raises:
        HTTPException: 404 if the project doesn't exist; 500 if the README
            Writer agent call fails.
    """
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
                "folder_readme": await _get_latest_folder_readme(db, project_id),
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
            "folder_readme": await _get_latest_folder_readme(db, project_id),
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
    """Save the author's manual edits to the project README and mark it reviewed.

    Args:
        project_id: CodeProject whose ``readme_md`` is being edited.
        request: Current request; the form body carries ``readme_md``.
        db: Async DB session (injected).

    Returns:
        The re-rendered ``partials/project_agent_panel.html`` fragment.

    Raises:
        HTTPException: 404 if the project doesn't exist.
    """
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
            "folder_readme": await _get_latest_folder_readme(db, project_id),
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
    """Push the reviewed project README to GitHub as README.md.

    Args:
        project_id: CodeProject whose README is being pushed.
        request: Current request, forwarded to the template response.
        db: Async DB session (injected).

    Returns:
        The re-rendered ``partials/project_agent_panel.html`` fragment,
        either with an error (README not yet reviewed) or a success toast.

    Raises:
        HTTPException: 404 if the project doesn't exist; 502 if the GitHub
            push itself fails.
    """
    project = await db.get(CodeProject, project_id)
    if not project:
        raise HTTPException(status_code=404)

    if project.readme_status not in (ReadmeStatus.REVIEWED, ReadmeStatus.APPROVED):
        return templates.TemplateResponse(
            "partials/project_agent_panel.html",
            {
                "request": request,
                "project": project,
                "folder_readme": await _get_latest_folder_readme(db, project_id),
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
            "folder_readme": await _get_latest_folder_readme(db, project_id),
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
                "folder_readme": await _get_latest_folder_readme(db, project_id),
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
            "folder_readme": await _get_latest_folder_readme(db, project_id),
            "toast": f"README queued (run: {run_id[:8]}…). Badges update when done.",
        },
    )
