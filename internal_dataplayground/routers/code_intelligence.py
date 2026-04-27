# routers/code_intelligence.py
"""
Code Intelligence Module — v2 fixes:
  1. Sync Files no longer wipes the file tree — it reloads project_detail
     which preserves the tree. The Sync button now targets #ci-detail-panel
     correctly without an HTMX swap that clears the tree.

  2. Folder README is stored separately (folder_readme_md + folder_readme_path
     on CodeProject) and never overwrites readme_md. The /trigger-readme endpoint
     now stores folder README directly to the DB; the polling endpoint returns it.

  3. GET /code-intel accepts ?project_id= query param to auto-select a project
     on page load. The template receives `preload_project_id` and JS fires
     openProject() automatically.

  4. Removed the duplicate /trigger-readme route definition that caused the
     "Internal Server Error" on Full Project README generation.
"""

import logging
from datetime import datetime
from typing import Optional
import httpx

from fastapi import APIRouter, Depends, HTTPException, Request, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from database import get_db
from models import CodeProject, CodeFile, ReadmeStatus, CommentedStatus, ImprovementStatus
from services.github_service import (
    pull_file_content,
    list_repo_files,
    push_file_content,
    get_file_sha,
)
from airflow.agents.blog_agents import (
    agent_code_narrator,
    agent_code_commenter,
    agent_code_improver,
    agent_readme_writer,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/code-intel", tags=["Code Intelligence"])
templates = Jinja2Templates(directory="templates")

CODE_NARRATE_DAG  = "life_os_code_narrate"
CODE_COMMENT_DAG  = "life_os_code_comment"
CODE_IMPROVE_DAG  = "life_os_code_improve"
README_WRITER_DAG = "life_os_readme_writer"


async def _trigger_airflow(dag_id: str, conf: dict) -> str:
    """Trigger an Airflow DAG and return the run_id."""
    import base64
    from gcp_secrets import get_key
    password = get_key("Airflow-Admin-Password")
    token = base64.b64encode(f"admin:{password}".encode()).decode()
    headers = {"Authorization": f"Basic {token}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"http://airflow-webserver:8080/api/v1/dags/{dag_id}/dagRuns",
            headers=headers,
            json={"conf": conf},
        )

        # Log the actual error body instead of just raising blindly
        if resp.status_code not in (200, 201, 409):  # 409 = already running, still OK
            log.error("Airflow trigger failed: %s — %s", resp.status_code, resp.text)
            resp.raise_for_status()
        
        data = resp.json()
        return data.get("dag_run_id", "unknown")


# ── PROJECT MANAGEMENT ─────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def project_list_ui(
    request: Request,
    db: AsyncSession = Depends(get_db),
    project_id: Optional[int] = Query(default=None),  # FIX 3: query param preload
):
    """
    Main Code Intelligence page.
    Pass ?project_id=N to auto-open a project's detail panel on load.
    """
    result = await db.execute(select(CodeProject).order_by(CodeProject.project_name))
    projects = result.scalars().all()
    return templates.TemplateResponse("code_intelligence.html", {
        "request": request,
        "projects": projects,
        "active_module": "code_intel",
        "preload_project_id": project_id,  # passed to template for JS auto-select
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
        {"request": request, "projects": projects, "toast": f"Project '{project_name}' created."},
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
    """
    FIX 1: Pulls file tree from GitHub and syncs CodeFile rows.
    Returns project_detail partial (not a blank panel) so the tree
    is immediately rendered after sync without losing context.
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
        # On error, still return the project detail with an error message
        # so the tree is not wiped
        return templates.TemplateResponse(
            "partials/project_detail.html",
            {"request": request, "project": project,
             "error": f"GitHub API error: {exc}"},
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

    # Reload project with fresh files list
    await db.refresh(project)
    # Expire and reload to get updated files relationship
    from sqlalchemy import inspect
    db.expire(project)
    project = await db.get(CodeProject, project_id)

    return templates.TemplateResponse(
        "partials/project_detail.html",
        {"request": request, "project": project,
         "toast": f"Synced. {new_count} new file(s) added, {len(existing_paths)} already tracked."},
    )


# ── README MANAGEMENT (project-level) ─────────────────────────────────────────

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
            {"request": request, "project": project,
             "error": "No file narrations found. Narrate at least one file first."},
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
        {"request": request, "project": project,
         "toast": "README generated. Review and edit before pushing."},
    )


@router.patch("/projects/{project_id}/readme", response_class=HTMLResponse)
async def save_readme_edits(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Saves manual edits to the project README and marks it as reviewed."""
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
        {"request": request, "project": project,
         "toast": "README saved and marked as reviewed."},
    )


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
            {"request": request, "project": project,
             "error": "Mark the README as reviewed before pushing."},
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
        {"request": request, "project": project,
         "toast": f"README pushed to GitHub at {readme_path} ✓"},
    )


# ── README TRIGGER (Airflow) ───────────────────────────────────────────────────
# FIX 4: Single, non-duplicated /trigger-readme endpoint.
# FIX 2: Folder README stored in folder_readme_md / folder_readme_path columns,
#         never touching readme_md on the project.

@router.post("/projects/{project_id}/trigger-readme", response_class=HTMLResponse)
async def trigger_readme_dag(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Triggers the README Writer DAG in Airflow.

    - No folder_path  → full project README queued.
                        Returns project_detail partial with toast.
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
        run_id = await _trigger_airflow(README_WRITER_DAG, conf)
    except Exception as exc:
        if folder_path:
            return JSONResponse({"error": str(exc)}, status_code=502)
        return templates.TemplateResponse(
            "partials/project_agent_panel.html",
            {"request": request, "project": project,
             "error": f"Airflow unreachable: {exc}"},
        )

    if folder_path:
        return JSONResponse({
            "run_id": run_id,
            "status": "queued",
            "folder_path": folder_path,
        })
    else:
        return templates.TemplateResponse(
            "partials/project_agent_panel.html",
            {
                "request": request,
                "project": project,
                "toast": f"README queued (run: {run_id[:8]}…). Badges update when done.",
            },
        )


# ── FOLDER README — save / retrieve ────────────────────────────────────────────
# FIX 2: Dedicated endpoints for folder README that never touch readme_md.

@router.patch("/projects/{project_id}/folder-readme", response_class=JSONResponse)
async def save_folder_readme(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Saves a folder-scoped README to folder_readme_md WITHOUT touching readme_md.
    Called from the Airflow DAG callback OR from 'Save as Project README' in UI.

    Body JSON: {"content": str, "folder_path": str, "save_as_project": bool}
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
        # User explicitly chose "Save as Project README"
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
    """
    Returns the stored folder README content + which folder it's for.
    Called by the frontend after polling detects the DAG completed.
    """
    project = await db.get(CodeProject, project_id)
    if not project or not project.folder_readme_md:
        raise HTTPException(status_code=404, detail="No folder README available")

    return {
        "content": project.folder_readme_md,
        "folder_path": project.folder_readme_path or "",
        "generated_at": str(project.folder_readme_generated_at) if project.folder_readme_generated_at else None,
    }


# ── FILE OPERATIONS ────────────────────────────────────────────────────────────

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


@router.post("/files/{file_id}/pull", response_class=HTMLResponse)
async def pull_file(
    file_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Pulls latest raw code from GitHub. Returns JSON for batch pulls, HTML for single."""
    code_file = await db.get(CodeFile, file_id)
    if not code_file:
        raise HTTPException(status_code=404)

    project = await db.get(CodeProject, code_file.project_id)

    try:
        content, sha = await pull_file_content(project.github_repo, code_file.github_path)
    except Exception as exc:
        # Check if this is a batch pull (Accept: application/json header)
        accept = request.headers.get("accept", "")
        if "application/json" in accept:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=502)
        raise HTTPException(status_code=502, detail=f"GitHub pull failed: {exc}")

    sha_changed = sha != code_file.github_sha
    code_file.raw_code = content
    code_file.github_sha = sha
    code_file.code_pulled_at = datetime.utcnow()
    await db.commit()
    await db.refresh(code_file)

    # If called from batch pull (JS fetch), return minimal JSON
    accept = request.headers.get("accept", "")
    if "application/json" in accept:
        return JSONResponse({
            "ok": True,
            "sha_changed": sha_changed,
            "chars": len(content),
        })

    toast = f"Code pulled ({len(content)} chars)."
    if sha_changed and code_file.narration:
        toast += " ⚠ Code changed — narration may be stale."

    return templates.TemplateResponse(
        "partials/code_file_detail.html",
        {"request": request, "file": code_file, "toast": toast},
    )


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
            {"request": request, "file": code_file,
             "error": "Pull the code from GitHub first."},
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
        run_id = await _trigger_airflow(CODE_NARRATE_DAG, {"file_ids": file_ids, "project_id": project_id})
        return JSONResponse({"run_id": run_id, "status": "triggered", "count": len(file_ids)})
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Airflow unreachable: {exc}")


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
        {"request": request, "file": code_file,
         "toast": "Commented version generated. Review before pushing."},
    )


@router.post("/batch/comment")
async def trigger_batch_comment(request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    file_ids = data.get("file_ids", [])
    if not file_ids:
        raise HTTPException(status_code=422, detail="file_ids required")
    try:
        run_id = await _trigger_airflow(CODE_COMMENT_DAG, {"file_ids": file_ids})
        return JSONResponse({"run_id": run_id, "status": "triggered", "count": len(file_ids)})
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Airflow unreachable: {exc}")


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
        {"request": request, "file": code_file,
         "toast": "Improvement report generated. Review the suggestions."},
    )


@router.post("/batch/improve")
async def trigger_batch_improve(request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    file_ids = data.get("file_ids", [])
    if not file_ids:
        raise HTTPException(status_code=422, detail="file_ids required")
    try:
        run_id = await _trigger_airflow(CODE_IMPROVE_DAG, {"file_ids": file_ids})
        return JSONResponse({"run_id": run_id, "status": "triggered", "count": len(file_ids)})
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Airflow unreachable: {exc}")


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
            {"request": request, "file": code_file,
             "error": "Mark the commented version as reviewed before pushing."},
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
        {"request": request, "file": code_file,
         "toast": f"Commented {code_file.file_name} pushed to GitHub ✓"},
    )


@router.get("/projects/{project_id}/detail", response_class=HTMLResponse)
async def project_detail(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(CodeProject, project_id)
    if not project:
        raise HTTPException(status_code=404)
    return templates.TemplateResponse(
        "partials/project_detail.html",
        {"request": request, "project": project},
    )


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
        {"request": request, "file": code_file,
         "toast": "Commented code marked as reviewed. Ready to push to GitHub."},
    )


@router.get("/projects/{project_id}/file-statuses")
async def get_file_statuses(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns lightweight status for every file in the project.
    Used by the frontend to refresh badges in-place without reloading the tree.
    """
    result = await db.execute(
        text("""
            SELECT
                id,
                raw_code IS NOT NULL         AS has_code,
                narration IS NOT NULL        AS narrated,
                (narration IS NOT NULL AND code_pulled_at IS NOT NULL
                 AND code_pulled_at > narration_generated_at) AS narrate_stale,
                commented_status,
                improvement_status
            FROM code_files
            WHERE project_id = :pid
            ORDER BY github_path
        """),
        {"pid": project_id}
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
async def project_status(project_id: int, db: AsyncSession = Depends(get_db)):
    """
    Lightweight polling endpoint. Returns counts + last_updated timestamp.
    Also returns folder_readme_updated flag when a folder README DAG completes.
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
        {"pid": project_id}
    )
    row = result.mappings().one()

    project = await db.get(CodeProject, project_id)
    folder_readme_updated = False
    if project and project.folder_readme_generated_at and row["last_updated"]:
        folder_readme_updated = (
            project.folder_readme_generated_at > row["last_updated"]
        )

    return {
        "total": row["total"],
        "narrated": row["narrated"] or 0,
        "commented": row["commented"] or 0,
        "improved": row["improved"] or 0,
        "last_updated": str(row["last_updated"]) if row["last_updated"] else None,
        "folder_readme_updated": folder_readme_updated,
        "folder_readme_path": project.folder_readme_path if project else None,
        "readme_status": project.readme_status.value if project else "none",
    }


@router.get("/projects/{project_id}/readme-content")
async def get_readme_content(
    project_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Returns the project-level readme_md and generation metadata."""
    project = await db.get(CodeProject, project_id)
    if not project or not project.readme_md:
        raise HTTPException(status_code=404, detail="No README available")

    return {
        "content": project.readme_md,
        "folder_path": None,
        "generated_at": str(project.readme_generated_at) if project.readme_generated_at else None,
    }
