# routers/ci_folder_readme.py
"""
Code Intelligence — Folder-Scoped README Storage

Split out of ci_readme.py (WO#24 Part A) — holds the per-folder README
lookup helpers (shared with ci_projects.py) and the folder-README
save/retrieve endpoints, as a cohesive concern distinct from the
project-level README generate/save/push/trigger flow that stays in
ci_readme.py.

Endpoints:
  PATCH /code-intel/projects/{id}/folder-readme → Save folder-scoped README
  GET   /code-intel/projects/{id}/folder-readme → Retrieve folder README content
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.code_intel.models import (
    CodeProject, ReadmeStatus, FolderReadme, FolderReadmeStatus,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/code-intel", tags=["Code Intelligence"])


# ── Folder README lookup helpers ───────────────────────────────────────────────
# Shared by the folder-readme endpoints below and by routers/ci_projects.py
# (project_detail, sync_files_from_github, project_status) and
# routers/ci_readme.py (generate_readme, save_readme_edits, push_readme,
# trigger_readme_dag), all of which need to know about a project's
# per-folder READMEs without duplicating the query.

async def _get_folder_readme(
    db: AsyncSession, project_id: int, folder_path: str
) -> Optional[FolderReadme]:
    """Fetch the FolderReadme row for one exact (project, folder) pair.

    Args:
        db: Async DB session.
        project_id: Owning CodeProject's primary key.
        folder_path: Repo-relative folder path, e.g. "internal_dataplayground/routers".

    Returns:
        The matching FolderReadme row, or None if that folder has never had
        a README generated for it.
    """
    result = await db.execute(
        select(FolderReadme).where(
            FolderReadme.project_id == project_id,
            FolderReadme.folder_path == folder_path,
        )
    )
    return result.scalar_one_or_none()


async def _get_latest_folder_readme(
    db: AsyncSession, project_id: int
) -> Optional[FolderReadme]:
    """Fetch the most recently generated folder README for a project, if any.

    Used to re-populate the folder-README panel when the project detail /
    agent panel partials are re-rendered (e.g. after a sync or a project
    README save) so a previously generated folder README doesn't appear to
    vanish just because the page/panel reloaded.

    Args:
        db: Async DB session.
        project_id: Owning CodeProject's primary key.

    Returns:
        The FolderReadme row with the newest `readme_generated_at` that has
        content, or None if no folder README has ever been generated for
        this project.
    """
    result = await db.execute(
        select(FolderReadme)
        .where(
            FolderReadme.project_id == project_id,
            FolderReadme.readme_md.is_not(None),
        )
        .order_by(FolderReadme.readme_generated_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


# ── Folder README — save / retrieve ───────────────────────────────────────────

@router.patch("/projects/{project_id}/folder-readme", response_class=JSONResponse)
async def save_folder_readme(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Save a folder-scoped README, persisted per-folder in FolderReadme.

    Upserts by ``(project_id, folder_path)`` — a project can hold one
    persisted README per folder (e.g. one for ``routers/`` and a separate
    one for ``templates/`` at the same time); saving a new folder never
    overwrites a different folder's stored README, and re-saving the same
    folder simply updates that folder's row.

    Called from the Airflow DAG callback after a folder-scoped README
    Writer run, OR from "Save as Project README" in the UI.

    Args:
        project_id: Owning CodeProject's primary key.
        request: Current request; the JSON body carries ``content``,
            ``folder_path``, and optional ``save_as_project``.
        db: Async DB session (injected).

    Returns:
        ``{"ok": True, "saved_as_project": bool}``.

    Raises:
        HTTPException: 404 if the project doesn't exist; 422 if
            ``folder_path`` is missing (a folder README always belongs to
            exactly one folder — there's no project-wide "default" row).
    """
    data = await request.json()
    content = data.get("content", "").strip()
    folder_path = data.get("folder_path", "").strip()
    save_as_project = bool(data.get("save_as_project", False))

    project = await db.get(CodeProject, project_id)
    if not project:
        raise HTTPException(status_code=404)

    if not folder_path:
        raise HTTPException(status_code=422, detail="folder_path is required")

    folder_readme = await _get_folder_readme(db, project_id, folder_path)
    if folder_readme is None:
        folder_readme = FolderReadme(
            project_id=project_id,
            folder_path=folder_path,
            folder_display_name=folder_path.split("/")[-1] or folder_path,
        )
        db.add(folder_readme)

    folder_readme.readme_md = content
    folder_readme.readme_generated_at = datetime.utcnow()
    folder_readme.status = FolderReadmeStatus.DRAFT

    if save_as_project:
        project.readme_md = content
        project.readme_status = ReadmeStatus.REVIEWED
        project.readme_generated_at = datetime.utcnow()

    await db.commit()
    return {"ok": True, "saved_as_project": save_as_project}


@router.get("/projects/{project_id}/folder-readme")
async def get_folder_readme(
    project_id: int,
    folder_path: str,
    db: AsyncSession = Depends(get_db),
):
    """Return the stored README for one folder of a project.

    Args:
        project_id: Owning CodeProject's primary key.
        folder_path: Which folder's README to retrieve. Required — a
            project can have several persisted folder READMEs, so there is
            no longer a single implicit "the" folder README to fall back to.
        db: Async DB session (injected).

    Returns:
        ``{"content": str, "folder_path": str, "generated_at": str | None}``.

    Raises:
        HTTPException: 404 if no README has been generated for that folder yet.
    """
    folder_readme = await _get_folder_readme(db, project_id, folder_path)
    if not folder_readme or not folder_readme.readme_md:
        raise HTTPException(status_code=404, detail="No folder README available")

    return {
        "content": folder_readme.readme_md,
        "folder_path": folder_readme.folder_path,
        "generated_at": (
            str(folder_readme.readme_generated_at)
            if folder_readme.readme_generated_at else None
        ),
    }
