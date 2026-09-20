# routers/ci_status.py
"""
Code Intelligence — Frontend Polling / Status API

Split out of ci_projects.py (WO#24 Part C) — holds the read-only
badge-refresh and batch-completion polling endpoints the frontend calls
on an interval, as a cohesive concern distinct from project CRUD.

Endpoints:
  GET  /code-intel/projects/{id}/file-statuses → Badge refresh JSON
  GET  /code-intel/projects/{id}/status        → Polling status JSON
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domains.code_intel.models import CodeProject
from domains.code_intel.routers.ci_folder_readme import _get_folder_readme

log = logging.getLogger(__name__)

router = APIRouter(prefix="/code-intel", tags=["Code Intelligence"])


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
