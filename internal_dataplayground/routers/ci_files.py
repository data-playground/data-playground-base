# routers/ci_files.py
"""
Code Intelligence — File Operations and Batch Triggers

Endpoints:
  GET   /code-intel/files/{id}               → File detail partial (HTMX drawer)
  POST  /code-intel/files/{id}/pull          → Pull raw code from GitHub
  POST  /code-intel/files/{id}/narrate       → Run Code Narrator (inline, single file)
  POST  /code-intel/files/{id}/comment       → Run Code Commenter (inline, single file)
  POST  /code-intel/files/{id}/improve       → Run Code Improver (inline, single file)
  POST  /code-intel/files/{id}/push-comments → Push commented code to GitHub
  PATCH /code-intel/files/{id}/comment-status → Update comment review status
  POST  /code-intel/batch/narrate            → Trigger narrate DAG for multiple files
  POST  /code-intel/batch/comment            → Trigger comment DAG for multiple files
  POST  /code-intel/batch/improve            → Trigger improve DAG for multiple files
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models import CodeFile, CodeProject, CommentedStatus, ImprovementStatus
from services.airflow_service import trigger_airflow
from services.github_service import pull_file_content, push_file_content
from airflow.agents.blog_agents import (
    agent_code_narrator,
    agent_code_commenter,
    agent_code_improver,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/code-intel", tags=["Code Intelligence"])
templates = Jinja2Templates(directory="templates")

# ── DAG identifiers ────────────────────────────────────────────────────────────
CODE_NARRATE_DAG = "life_os_code_narrate"
CODE_COMMENT_DAG = "life_os_code_comment"
CODE_IMPROVE_DAG = "life_os_code_improve"


# ── File detail ────────────────────────────────────────────────────────────────

@router.get("/files/{file_id}", response_class=HTMLResponse)
async def file_detail(
    file_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    code_file = await db.get(CodeFile, file_id)
    if not code_file:
        raise HTTPException(status_code=404)
    return templates.TemplateResponse(
        "partials/code_file_detail.html",
        {"request": request, "file": code_file},
    )


# ── Pull from GitHub ───────────────────────────────────────────────────────────

@router.post("/files/{file_id}/pull", response_class=HTMLResponse)
async def pull_file(
    file_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Pulls latest raw code from GitHub.
    Returns JSON when called from the batch pull JS (Accept: application/json),
    otherwise returns the file detail HTML partial.
    """
    code_file = await db.get(CodeFile, file_id)
    if not code_file:
        raise HTTPException(status_code=404)

    project = await db.get(CodeProject, code_file.project_id)
    accept = request.headers.get("accept", "")

    try:
        content, sha = await pull_file_content(project.github_repo, code_file.github_path)
    except Exception as exc:
        if "application/json" in accept:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
        raise HTTPException(status_code=502, detail=f"GitHub pull failed: {exc}")

    sha_changed = sha != code_file.github_sha
    code_file.raw_code = content
    code_file.github_sha = sha
    code_file.code_pulled_at = datetime.utcnow()
    await db.commit()
    await db.refresh(code_file)

    if "application/json" in accept:
        return JSONResponse({"ok": True, "sha_changed": sha_changed, "chars": len(content)})

    toast = f"Code pulled ({len(content)} chars)."
    if sha_changed and code_file.narration:
        toast += " ⚠ Code changed — narration may be stale."

    return templates.TemplateResponse(
        "partials/code_file_detail.html",
        {"request": request, "file": code_file, "toast": toast},
    )


# ── Inline single-file agents ──────────────────────────────────────────────────

@router.post("/files/{file_id}/narrate", response_class=HTMLResponse)
async def narrate_file(
    file_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    code_file = await db.get(CodeFile, file_id)
    if not code_file:
        raise HTTPException(status_code=404)
    if not code_file.raw_code:
        return templates.TemplateResponse(
            "partials/code_file_detail.html",
            {"request": request, "file": code_file, "error": "Pull the code from GitHub first."},
        )

    project = await db.get(CodeProject, code_file.project_id)

    try:
        narration = agent_code_narrator(
            code_content=code_file.raw_code,
            file_name=code_file.file_name,
            readme_context=project.readme_md or project.description or "",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Code Narrator failed: {exc}")

    code_file.narration = narration
    code_file.narration_generated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(code_file)

    return templates.TemplateResponse(
        "partials/code_file_detail.html",
        {"request": request, "file": code_file, "toast": "Narration generated ✓"},
    )


@router.post("/files/{file_id}/comment", response_class=HTMLResponse)
async def comment_file(
    file_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    code_file = await db.get(CodeFile, file_id)
    if not code_file or not code_file.raw_code:
        raise HTTPException(status_code=404, detail="File not found or no code pulled.")

    try:
        commented = agent_code_commenter(
            code_content=code_file.raw_code,
            file_name=code_file.file_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Code Commenter failed: {exc}")

    code_file.commented_code = commented
    code_file.commented_generated_at = datetime.utcnow()
    code_file.commented_status = CommentedStatus.GENERATED
    await db.commit()
    await db.refresh(code_file)

    return templates.TemplateResponse(
        "partials/code_file_detail.html",
        {
            "request": request,
            "file": code_file,
            "toast": "Commented version generated. Review before pushing.",
        },
    )


@router.post("/files/{file_id}/improve", response_class=HTMLResponse)
async def improve_file(
    file_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    code_file = await db.get(CodeFile, file_id)
    if not code_file or not code_file.raw_code:
        raise HTTPException(status_code=404, detail="File not found or no code pulled.")

    try:
        notes = agent_code_improver(
            code_content=code_file.raw_code,
            file_name=code_file.file_name,
            narration=code_file.narration or "",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Code Improver failed: {exc}")

    code_file.improvement_notes = notes
    code_file.improvement_generated_at = datetime.utcnow()
    code_file.improvement_status = ImprovementStatus.GENERATED
    await db.commit()
    await db.refresh(code_file)

    return templates.TemplateResponse(
        "partials/code_file_detail.html",
        {
            "request": request,
            "file": code_file,
            "toast": "Improvement report generated. Review the suggestions.",
        },
    )


# ── Push commented code to GitHub ─────────────────────────────────────────────

@router.post("/files/{file_id}/push-comments", response_class=HTMLResponse)
async def push_commented_file(
    file_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    code_file = await db.get(CodeFile, file_id)
    if not code_file:
        raise HTTPException(status_code=404)

    if code_file.commented_status != CommentedStatus.REVIEWED:
        return templates.TemplateResponse(
            "partials/code_file_detail.html",
            {
                "request": request,
                "file": code_file,
                "error": "Mark the commented version as reviewed before pushing.",
            },
        )

    project = await db.get(CodeProject, code_file.project_id)

    try:
        new_sha = await push_file_content(
            repo=project.github_repo,
            path=code_file.github_path,
            content=code_file.commented_code or "",
            sha=code_file.github_sha,
            commit_message=f"docs: add comments to {code_file.file_name} via Life OS",
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"GitHub push failed: {exc}")

    code_file.github_sha = new_sha
    code_file.commented_status = CommentedStatus.PUSHED
    await db.commit()
    await db.refresh(code_file)

    return templates.TemplateResponse(
        "partials/code_file_detail.html",
        {
            "request": request,
            "file": code_file,
            "toast": f"Commented {code_file.file_name} pushed to GitHub ✓",
        },
    )


# ── Comment status patch ───────────────────────────────────────────────────────

@router.patch("/files/{file_id}/comment-status", response_class=HTMLResponse)
async def update_comment_status(
    file_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    form = await request.form()
    status_str = str(form.get("commented_status", "")).strip()

    try:
        new_status = CommentedStatus(status_str)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid status: {status_str}")

    code_file = await db.get(CodeFile, file_id)
    if not code_file:
        raise HTTPException(status_code=404)

    code_file.commented_status = new_status
    await db.commit()
    await db.refresh(code_file)

    return templates.TemplateResponse(
        "partials/code_file_detail.html",
        {
            "request": request,
            "file": code_file,
            "toast": "Commented code marked as reviewed. Ready to push to GitHub.",
        },
    )


# ── Batch Airflow triggers ─────────────────────────────────────────────────────

@router.post("/batch/narrate")
async def trigger_batch_narrate(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    data = await request.json()
    file_ids = data.get("file_ids", [])
    project_id = data.get("project_id")
    if not file_ids:
        raise HTTPException(status_code=422, detail="file_ids required")
    try:
        run_id = await trigger_airflow(
            CODE_NARRATE_DAG, {"file_ids": file_ids, "project_id": project_id}
        )
        return JSONResponse({"run_id": run_id, "status": "triggered", "count": len(file_ids)})
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Airflow unreachable: {exc}")


@router.post("/batch/comment")
async def trigger_batch_comment(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    data = await request.json()
    file_ids = data.get("file_ids", [])
    if not file_ids:
        raise HTTPException(status_code=422, detail="file_ids required")
    try:
        run_id = await trigger_airflow(CODE_COMMENT_DAG, {"file_ids": file_ids})
        return JSONResponse({"run_id": run_id, "status": "triggered", "count": len(file_ids)})
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Airflow unreachable: {exc}")


@router.post("/batch/improve")
async def trigger_batch_improve(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    data = await request.json()
    file_ids = data.get("file_ids", [])
    if not file_ids:
        raise HTTPException(status_code=422, detail="file_ids required")
    try:
        run_id = await trigger_airflow(CODE_IMPROVE_DAG, {"file_ids": file_ids})
        return JSONResponse({"run_id": run_id, "status": "triggered", "count": len(file_ids)})
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Airflow unreachable: {exc}")
