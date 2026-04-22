# routers/code_intelligence.py
"""
Code Intelligence Module

Manages CodeProjects (GitHub repo scopes) and CodeFiles (individual scripts).
Provides on-demand agent triggers for:
  - Code Narrator    → narration (feeds blog agents)
  - Code Commenter   → commented_code (review + push to GitHub)
  - Code Improver    → improvement_notes (review only)
  - README Writer    → readme_md on CodeProject (review + push to GitHub)

All agent outputs require human review before any GitHub push.

Endpoints:
  GET    /code-intel                              → Project list UI
  POST   /code-intel/projects                     → Create a new project
  DELETE /code-intel/projects/{id}                → Delete project + all files
  POST   /code-intel/projects/{id}/sync           → Pull file list from GitHub
  POST   /code-intel/projects/{id}/generate-readme → Run README Writer
  PATCH  /code-intel/projects/{id}/readme         → Save edits to README
  POST   /code-intel/projects/{id}/push-readme    → Push approved README to GitHub
  GET    /code-intel/projects/{id}/files          → File list partial
  POST   /code-intel/files/{id}/pull              → Pull latest code from GitHub
  POST   /code-intel/files/{id}/narrate           → Run Code Narrator
  POST   /code-intel/files/{id}/comment           → Run Code Commenter
  POST   /code-intel/files/{id}/improve           → Run Code Improver
  POST   /code-intel/files/{id}/push-comments     → Push commented code to GitHub
  GET    /code-intel/files/{id}                   → File detail partial
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

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
            "http://airflow-webserver:8080/api/v1/dagRuns".replace("dagRuns", f"dags/{dag_id}/dagRuns"),
            headers=headers,
            json={"conf": conf},
        )
        resp.raise_for_status()
        return resp.json().get("dag_run_id", "unknown")

# ── PROJECT MANAGEMENT ─────────────────────────────────────────────────────────

@router.get("", response_class=HTMLResponse)
async def project_list_ui(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(CodeProject).order_by(CodeProject.project_name))
    projects = result.scalars().all()
    return templates.TemplateResponse("code_intelligence.html", {
        "request": request,
        "projects": projects,
        "active_module": "code_intel",
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
    """
    Creates a new CodeProject.
    github_base_path examples:
      ""  or omit  → whole repo
      "internal_dataplayground"          → app subfolder
      "internal_dataplayground/routers"  → just routers
    """
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
    Pulls the file tree from GitHub for the project's repo + base_path.
    Creates CodeFile rows for any new .py files discovered.
    Does NOT pull file content yet — use /files/{id}/pull for that.
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
        raise HTTPException(status_code=502, detail=f"GitHub API error: {exc}")

    # Get existing tracked paths to avoid duplicates
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

    # Reload project with files
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
    """
    Runs README Writer agent using all file narrations in the project.
    Requires at least some files to have narrations generated first.
    """
    project = await db.get(CodeProject, project_id)
    if not project:
        raise HTTPException(status_code=404)

    # Collect narrations from all tracked files
    narrated_files = [f for f in project.files if f.narration]
    if not narrated_files:
        return templates.TemplateResponse(
            "partials/project_detail.html",
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
        "partials/project_detail.html",
        {"request": request, "project": project,
         "toast": "README generated. Review and edit before pushing."},
    )


@router.patch("/projects/{project_id}/readme", response_class=HTMLResponse)
async def save_readme_edits(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Saves manual edits to the README and marks it as reviewed."""
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
        "partials/project_detail.html",
        {"request": request, "project": project,
         "toast": "README saved and marked as reviewed."},
    )


@router.post("/projects/{project_id}/push-readme", response_class=HTMLResponse)
async def push_readme(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Pushes the approved README.md to GitHub.
    README is placed at: {github_base_path}/README.md
    (or repo root if base_path is empty)
    """
    project = await db.get(CodeProject, project_id)
    if not project:
        raise HTTPException(status_code=404)

    if project.readme_status not in (ReadmeStatus.REVIEWED, ReadmeStatus.APPROVED):
        return templates.TemplateResponse(
            "partials/project_detail.html",
            {"request": request, "project": project,
             "error": "Mark the README as reviewed before pushing."},
        )

    # Determine target path in repo
    base = (project.github_base_path or "").rstrip("/")
    readme_path = f"{base}/README.md" if base else "README.md"

    # Get current SHA of README on GitHub (needed for update, None for new file)
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
        "partials/project_detail.html",
        {"request": request, "project": project,
         "toast": f"README pushed to GitHub at {readme_path} ✓"},
    )

@router.post("/projects/{project_id}/trigger-readme", response_class=HTMLResponse)
async def trigger_readme(
    project_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    form = await request.form()
    folder_path = str(form.get("folder_path", "")).strip()
    conf = {"project_id": project_id}
    if folder_path:
        conf["folder_path"] = folder_path
    try:
        run_id = await _trigger_airflow(README_WRITER_DAG, conf)
        project = await db.get(CodeProject, project_id)
        label = f"/{folder_path.split('/')[-1]}" if folder_path else "project"
        return templates.TemplateResponse(
            "partials/project_detail.html",
            {"request": request, "project": project,
             "toast": f"README for {label} queued (run: {run_id[:8]}…). Refresh in ~30s."},
        )
    except Exception as exc:
        project = await db.get(CodeProject, project_id)
        return templates.TemplateResponse(
            "partials/project_detail.html",
            {"request": request, "project": project,
             "error": f"Airflow unreachable: {exc}"},
        )


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
    """Pulls latest raw code from GitHub and checks for SHA changes."""
    code_file = await db.get(CodeFile, file_id)
    if not code_file:
        raise HTTPException(status_code=404)

    project = await db.get(CodeProject, code_file.project_id)

    try:
        content, sha = await pull_file_content(project.github_repo, code_file.github_path)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"GitHub pull failed: {exc}")

    # Detect if code changed since last narration
    sha_changed = sha != code_file.github_sha

    code_file.raw_code = content
    code_file.github_sha = sha
    code_file.code_pulled_at = datetime.utcnow()
    await db.commit()
    await db.refresh(code_file)

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
    """Runs Code Narrator agent. Output feeds blog agents and README Writer."""
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

@router.post("/batch/narrate", response_class=HTMLResponse)
async def trigger_batch_narrate(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Triggers Airflow to narrate a batch of files. Body: {file_ids: [...], project_id: N}"""
    data = await request.json()
    file_ids = data.get("file_ids", [])
    project_id = data.get("project_id")
    if not file_ids:
        raise HTTPException(status_code=422, detail="file_ids required")
    try:
        run_id = await _trigger_airflow(CODE_NARRATE_DAG, {"file_ids": file_ids, "project_id": project_id})
        return HTMLResponse(f'{{"run_id": "{run_id}", "status": "triggered", "count": {len(file_ids)}}}', status_code=200)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Airflow unreachable: {exc}")



@router.post("/files/{file_id}/comment", response_class=HTMLResponse)
async def comment_file(
    file_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Runs Code Commenter agent. Output requires review before GitHub push."""
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

@router.post("/batch/comment", response_class=HTMLResponse)
async def trigger_batch_comment(request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    file_ids = data.get("file_ids", [])
    if not file_ids:
        raise HTTPException(status_code=422, detail="file_ids required")
    try:
        run_id = await _trigger_airflow(CODE_COMMENT_DAG, {"file_ids": file_ids})
        return HTMLResponse(f'{{"run_id": "{run_id}", "status": "triggered", "count": {len(file_ids)}}}')
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Airflow unreachable: {exc}")


@router.post("/files/{file_id}/improve", response_class=HTMLResponse)
async def improve_file(
    file_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Runs Code Improver agent.
    Output is a review report — never auto-applied or pushed.
    """
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

@router.post("/batch/improve", response_class=HTMLResponse)
async def trigger_batch_improve(request: Request, db: AsyncSession = Depends(get_db)):
    data = await request.json()
    file_ids = data.get("file_ids", [])
    if not file_ids:
        raise HTTPException(status_code=422, detail="file_ids required")
    try:
        run_id = await _trigger_airflow(CODE_IMPROVE_DAG, {"file_ids": file_ids})
        return HTMLResponse(f'{{"run_id": "{run_id}", "status": "triggered", "count": {len(file_ids)}}}')
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Airflow unreachable: {exc}")


@router.post("/files/{file_id}/push-comments", response_class=HTMLResponse)
async def push_commented_file(
    file_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Pushes the commented version of the file back to GitHub,
    replacing the original at the same path.
    Only allowed after status is REVIEWED.
    """
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

    code_file.github_sha = new_sha  # Update SHA after push
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
    """Loads the project detail partial into the right panel."""
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
    """
    Updates the commented_status field.
    Called from the UI when user clicks 'Mark Reviewed' on commented code.
    """
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


@router.get("/projects/{project_id}/status")
async def project_status(project_id: int, db: AsyncSession = Depends(get_db)):
    """
    Lightweight polling endpoint. Returns current narration/comment/improve
    counts so the UI can detect when a DAG run has updated files.
    """
    from sqlalchemy import text
    result = await db.execute(
        text("""
            SELECT
                COUNT(*) as total,
                SUM(narration IS NOT NULL) as narrated,
                SUM(commented_status != 'none') as commented,
                SUM(improvement_status != 'none') as improved,
                MAX(updated_at) as last_updated
            FROM code_files WHERE project_id = :pid
        """),
        {"pid": project_id}
    )
    row = result.mappings().one()
    return {
        "total": row["total"],
        "narrated": row["narrated"] or 0,
        "commented": row["commented"] or 0,
        "improved": row["improved"] or 0,
        "last_updated": str(row["last_updated"]) if row["last_updated"] else None,
    }