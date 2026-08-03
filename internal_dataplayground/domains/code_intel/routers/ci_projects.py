# routers/ci_projects.py
"""
Code Intelligence — Project Management

Endpoints:
  GET    /code-intel                              → Main page
  POST   /code-intel/projects                     → Create project
  DELETE /code-intel/projects/{id}                → Delete project
  POST   /code-intel/projects/{id}/sync           → Sync file tree from GitHub
  GET    /code-intel/projects/{id}/detail         → Project detail partial (HTMX)
  GET    /code-intel/projects/{id}/file-statuses  → Badge refresh JSON
  GET    /code-intel/projects/{id}/status         → Polling status JSON
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.code_intel.models import CodeFile, CodeProject
from services.github_service import list_repo_files
from core.templating import templates
from domains.code_intel.routers.ci_readme import (
    _get_folder_readme,
    _get_latest_folder_readme,
)

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


@router.get("/projects/{project_id}/file-statuses")
async def get_file_statuses(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Lightweight per-file status JSON used by the frontend to refresh
    tree badges in-place without reloading the full tree.
    """
    result = await db.execute(
        text("""
            SELECT
                id,
                raw_code IS NOT NULL                                              AS has_code,
                narration IS NOT NULL                                             AS narrated,
                (narration IS NOT NULL AND code_pulled_at IS NOT NULL
                 AND code_pulled_at > narration_generated_at)                     AS narrate_stale,
                commented_status,
                improvement_status
            FROM code_files
            WHERE project_id = :pid
            ORDER BY github_path
        """),
        {"pid": project_id},
    )
    rows = result.mappings().all()
    return [
        {
            "id": r["id"],
            "hasCode": bool(r["has_code"]),
            "narrated": bool(r["narrated"]),
            "narrateStale": bool(r["narrate_stale"]),
            "commentStatus": r["commented_status"],
            "improveStatus": r["improvement_status"],
        }
        for r in rows
    ]


@router.get("/projects/{project_id}/status")
async def project_status(
    project_id: int,
    folder_path: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Polling endpoint used after triggering a batch Airflow job.

    Returns file counts, a ``last_updated`` timestamp for detecting when
    narrate/comment/improve batches have finished, and (when ``folder_path``
    is supplied) whether that folder's README has been generated more
    recently than the current file state — the signal the frontend's
    "poll until folder README is ready" flow waits on.

    Args:
        project_id: CodeProject to report status for.
        folder_path: Optional. Which folder's README completion to check.
            Omit when polling for a project-wide narrate/comment/improve
            batch rather than a folder-scoped README generation.
        db: Async DB session (injected).

    Returns:
        A dict of counts plus ``folder_readme_updated`` (bool),
        ``folder_readme_path`` (echoes the input, for the frontend's
        convenience), and the project's README status badge value.

    Raises:
        HTTPException: 404 if the project doesn't exist.
    """
    result = await db.execute(
        text("""
            SELECT
                COUNT(*)                          AS total,
                SUM(narration IS NOT NULL)        AS narrated,
                SUM(commented_status != 'none')   AS commented,
                SUM(improvement_status != 'none') AS improved,
                MAX(updated_at)                   AS last_updated
            FROM code_files WHERE project_id = :pid
        """),
        {"pid": project_id},
    )
    row = result.mappings().one()

    project = await db.get(CodeProject, project_id)
    if not project:
        raise HTTPException(status_code=404)

    folder_readme_generated_at = None
    if folder_path:
        folder_readme = await _get_folder_readme(db, project_id, folder_path)
        if folder_readme:
            folder_readme_generated_at = folder_readme.readme_generated_at

    folder_readme_updated = False
    if folder_readme_generated_at and row["last_updated"]:
        try:
            folder_readme_updated = folder_readme_generated_at > row["last_updated"]
        except TypeError:
            folder_readme_updated = False

    return {
        "total": row["total"],
        "narrated": row["narrated"] or 0,
        "commented": row["commented"] or 0,
        "improved": row["improved"] or 0,
        "last_updated": str(row["last_updated"]) if row["last_updated"] else None,
        "folder_readme_updated": folder_readme_updated,
        "folder_readme_path": folder_path,
        "readme_status": project.readme_status.value if project else "none",
    }
