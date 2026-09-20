# routers/ci_batch.py
"""
Code Intelligence — Batch Airflow Triggers

Split out of ci_files.py (WO#24 Part B) — holds the multi-file batch
triggers, distinct from the inline single-file agent endpoints that stay
in ci_files.py.

Endpoints:
  POST  /code-intel/batch/narrate  → Trigger narrate DAG for multiple files
  POST  /code-intel/batch/comment  → Trigger comment DAG for multiple files
  POST  /code-intel/batch/improve  → Trigger improve DAG for multiple files
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from services.airflow_service import trigger_airflow

log = logging.getLogger(__name__)

router = APIRouter(prefix="/code-intel", tags=["Code Intelligence"])

# ── DAG identifiers ────────────────────────────────────────────────────────────
CODE_NARRATE_DAG = "life_os_code_narrate"
CODE_COMMENT_DAG = "life_os_code_comment"
CODE_IMPROVE_DAG = "life_os_code_improve"


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
