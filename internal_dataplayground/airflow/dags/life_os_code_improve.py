# airflow/dags/life_os_code_improve.py
"""
Code Improver DAG — triggered from LifeOS Code Intelligence UI.
Conf required: {"file_ids": [1, 2, 3]}

Changes vs original:
  - Added INTER_REQUEST_DELAY_SEC between file requests to stay under
    Cerebras's 30 RPM free tier limit. At 30 RPM you get one request
    every 2 seconds; we use 3s as a safe margin.
  - _cerebras() in blog_agents.py now handles 429s with automatic retry
    and backoff, so a burst that slips through is recovered gracefully.
  - Task retries increased to 2 with a 3-minute delay so the full DAG
    can recover from a temporary rate limit window without manual re-trigger.
  - Failed files are collected and reported at the end rather than silently
    skipped, making it clear in Airflow logs which files need a re-run.
"""
import sys
import logging
import time
from datetime import datetime, timedelta

sys.path.insert(0, '/opt/airflow/project')
sys.path.insert(0, '/opt/airflow/project/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

# ── RATE LIMIT CONFIGURATION ──────────────────────────────────────────────────
# Cerebras free tier: 30 RPM = 1 request every 2 seconds.
# We use 3 seconds as a conservative margin. Adjust downward if you upgrade
# to a paid Cerebras tier (which removes the RPM cap).
INTER_REQUEST_DELAY_SEC = 3

default_args = {
    "owner":             "life_os",
    "retries":           2,           # was 0 — allows recovery from transient 429 windows
    "retry_delay":       timedelta(minutes=3),
    "email_on_failure":  False,
}


def task_improve_files(**context):
    from dag_db import fetch_one, execute
    from agents.blog_agents import agent_code_improver

    file_ids = (context["dag_run"].conf or {}).get("file_ids", [])
    if not file_ids:
        raise ValueError("file_ids required in DAG conf")

    log.info(
        "Code Improver starting. %d file(s) to process. "
        "Inter-request delay: %ds (Cerebras 30 RPM free tier).",
        len(file_ids), INTER_REQUEST_DELAY_SEC,
    )

    failed_files = []

    for idx, file_id in enumerate(file_ids):
        # ── Throttle between requests ─────────────────────────────────────
        # Skip the delay before the very first request.
        if idx > 0:
            log.debug(
                "Waiting %ds before next request (%d/%d)...",
                INTER_REQUEST_DELAY_SEC, idx + 1, len(file_ids),
            )
            time.sleep(INTER_REQUEST_DELAY_SEC)

        file = fetch_one("SELECT * FROM code_files WHERE id = %s", (file_id,))
        if not file:
            log.warning("File %d not found, skipping", file_id)
            continue
        if not file.get("raw_code"):
            log.warning("File %d (%s) has no raw_code — pull it first, skipping",
                        file_id, file.get("file_name"))
            continue

        log.info("Improving %s (%d/%d)", file["file_name"], idx + 1, len(file_ids))
        try:
            notes = agent_code_improver(
                code_content=file["raw_code"],
                file_name=file["file_name"],
                narration=file.get("narration") or "",
            )
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            execute(
                """UPDATE code_files
                   SET improvement_notes = %s,
                       improvement_generated_at = %s,
                       improvement_status = %s,
                       updated_at = %s
                   WHERE id = %s""",
                (notes, now, "generated", now, file_id),
            )
            log.info("✓ Improved %s", file["file_name"])

        except Exception as exc:
            # _cerebras() already retried with backoff. If we still get here,
            # the error is persistent — record it and continue with other files
            # rather than failing the whole task.
            log.error(
                "✗ Failed to improve file %d (%s) after retries: %s",
                file_id, file.get("file_name", "?"), exc,
            )
            failed_files.append((file_id, file.get("file_name", "?"), str(exc)))

    # ── Final summary ─────────────────────────────────────────────────────
    succeeded = len(file_ids) - len(failed_files) - (len(file_ids) - len([
        f for f in file_ids
        if fetch_one("SELECT id FROM code_files WHERE id = %s", (f,))
    ]))
    log.info(
        "Code Improver complete. %d/%d files processed successfully.",
        len(file_ids) - len(failed_files), len(file_ids),
    )

    if failed_files:
        log.warning(
            "The following files failed and can be re-triggered individually:\n%s",
            "\n".join(f"  - id={fid} ({name}): {err}" for fid, name, err in failed_files),
        )
        # Raise so Airflow marks the task as failed and retries the whole
        # batch after retry_delay. The _cerebras retry handles per-request
        # 429s; this handles the case where the whole window was exhausted.
        raise RuntimeError(
            f"{len(failed_files)} file(s) failed. Check logs above for details."
        )


with DAG(
    dag_id="life_os_code_improve",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "code_intel"],
) as dag:
    PythonOperator(
        task_id="improve_files",
        python_callable=task_improve_files,
    )
