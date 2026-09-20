# routers/ci_projects.py
"""
Code Intelligence — Project Management (CRUD)

Endpoints:
  GET    /code-intel                              → Main page
  POST   /code-intel/projects                     → Create project
  DELETE /code-intel/projects/{id}                → Delete project
  POST   /code-intel/projects/{id}/sync           → Sync file tree from GitHub
  GET    /code-intel/projects/{id}/detail         → Project detail partial (HTMX)

Frontend polling/badge-refresh endpoints (GET .../file-statuses,
GET .../status) live in ci_status.py (WO#24 Part C) — split out as a
distinct concern from project CRUD.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Form, Query
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.code_intel.models import CodeFile, CodeProject
from services.github_service import list_repo_files
from core.templating import templates
from domains.code_intel.routers.ci_folder_readme import _get_latest_folder_readme

log = logging.getLogger(__name__)

router = APIRouter(prefix="/code-intel", tags=["Code Intelligence"])


@router.get("", response_class=HTMLResponse)
async def project_list_ui(
    request: Request,
    db: AsyncSession = Depends(get_db),
    project_id: Optional[int] = Query(default=None),
):
    """
    Main Code Intelligence page.
    Pass ?project_id=N to auto-open a project on load.
    """
    result = await db.execute(select(CodeProject).order_by(CodeProject.project_name))
    projects = result.scalars().all()
    return templates.TemplateResponse("code_intelligence.html", {
        "request": request,
        "projects": projects,
        "active_module": "code_intel",
        "preload_project_id": project_id,
    })


@router.post("/projects", response_class=HTMLResponse)
async def create_project(
    request: Request,
    project_name: str = Form(...),
    github_repo: str = Form(...),
    github_base_path: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    project = CodeProject(
        project_name=project_name.strip(),
        github_repo=github_repo.strip(),
        github_base_path=(github_base_path or "").strip() or None,
        description=(description or "").strip() or None,
    )
    db.add(project)
    await db.commit()
    await db.refresh(project)

    result = await db.execute(select(CodeProject).order_by(CodeProject.project_name))
    projects = result.scalars().all()
    return templates.TemplateResponse(
        "partials/project_list.html",
        {
            "request": request,
            "projects": projects,
            "toast": f"Project '{project_name}' created.",
        },
    )


@router.delete("/projects/{project_id}", response_class=HTMLResponse)
async def delete_project(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(CodeProject, project_id)
    if not project:
        raise HTTPException(status_code=404)
    await db.delete(project)
    await db.commit()

    result = await db.execute(select(CodeProject).order_by(CodeProject.project_name))
    projects = result.scalars().all()
    return templates.TemplateResponse(
        "partials/project_list.html",
        {"request": request, "projects": projects},
    )


@router.post("/projects/{project_id}/sync", response_class=HTMLResponse)
async def sync_files_from_github(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Pull the file tree from GitHub and upsert CodeFile rows.

    Returns the project_detail partial so the tree renders immediately
    after sync without losing panel context (selection state, the
    README Writer card, etc. are rebuilt fresh from this partial).

    Args:
        project_id: CodeProject to sync files for.
        request: Current request, forwarded to the template response.
        db: Async DB session (injected).

    Returns:
        The re-rendered ``partials/project_detail.html`` fragment, either
        with a GitHub error or a toast summarizing how many files were added.

    Raises:
        HTTPException: 404 if the project doesn't exist.
    """
    project = await db.get(CodeProject, project_id)
    if not project:
        raise HTTPException(status_code=404)

    try:
        github_files = await list_repo_files(
            repo=project.github_repo,
            base_path=project.github_base_path or "",
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "partials/project_detail.html",
            {
                "request": request,
                "project": project,
                "folder_readme": await _get_latest_folder_readme(db, project_id),
                "error": f"GitHub API error: {exc}",
            },
        )

    existing = await db.execute(
        select(CodeFile.github_path).where(CodeFile.project_id == project_id)
    )
    existing_paths = {r[0] for r in existing.all()}

    new_count = 0
    for f in github_files:
        if f["path"] not in existing_paths:
            code_file = CodeFile(
                project_id=project_id,
                file_name=f["name"],
                github_path=f["path"],
                github_sha=f["sha"],
            )
            db.add(code_file)
            new_count += 1

    await db.commit()

    # Expire and reload so the files relationship reflects the new rows
    db.expire(project)
    project = await db.get(CodeProject, project_id)

    return templates.TemplateResponse(
        "partials/project_detail.html",
        {
            "request": request,
            "project": project,
            "folder_readme": await _get_latest_folder_readme(db, project_id),
            "toast": f"Synced. {new_count} new file(s) added, {len(existing_paths)} already tracked.",
        },
    )


@router.get("/projects/{project_id}/detail", response_class=HTMLResponse)
async def project_detail(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Render the project detail panel (file tree + README Writer + agent cards).

    Args:
        project_id: CodeProject to render.
        request: Current request, forwarded to the template response.
        db: Async DB session (injected).

    Returns:
        The ``partials/project_detail.html`` fragment, pre-populated with
        the most recently generated folder README (if any) so it survives
        a panel reload instead of appearing to disappear.

    Raises:
        HTTPException: 404 if the project doesn't exist.
    """
    project = await db.get(CodeProject, project_id)
    if not project:
        raise HTTPException(status_code=404)
    return templates.TemplateResponse(
        "partials/project_detail.html",
        {
            "request": request,
            "project": project,
            "folder_readme": await _get_latest_folder_readme(db, project_id),
        },
    )
