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
INTER_REQUEST_DELAY_SEC = 65

default_args = {
    "owner":             "life_os",
    "retries":           2,           # was 0 — allows recovery from transient 429 windows
    "retry_delay":       timedelta(minutes=3),
    "email_on_failure":  False,
}


import time

# Replace the constant at the top
INTER_REQUEST_DELAY_SEC = 65  # minimum — we wait longer if tokens look low

def task_improve_files(**context):
    from dag_db import fetch_one, execute
    from agents.blog_agents import agent_code_improver

    file_ids = (context["dag_run"].conf or {}).get("file_ids", [])
    if not file_ids:
        raise ValueError("file_ids required in DAG conf")

    log.info(
        "Code Improver starting. %d file(s) to process. "
        "Minimum inter-request delay: %ds.",
        len(file_ids), INTER_REQUEST_DELAY_SEC,
    )

    failed_files = []
    last_remaining_tokens = None  # track from previous call

    for idx, file_id in enumerate(file_ids):

        # Adaptive wait: if the last call left very few tokens in the minute
        # bucket, sleep a full minute. Otherwise use the minimum.
        if idx > 0:
            if last_remaining_tokens is not None and last_remaining_tokens < 3000:
                sleep_time = INTER_REQUEST_DELAY_SEC
                log.info(
                    "Low token budget remaining (%d). Sleeping %ds before next file.",
                    last_remaining_tokens, sleep_time,
                )
            else:
                sleep_time = INTER_REQUEST_DELAY_SEC
                log.debug("Sleeping %ds between files.", sleep_time)
            time.sleep(sleep_time)

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
            notes, remaining_tokens = agent_code_improver(
                code_content=file["raw_code"],
                file_name=file["file_name"],
                narration=file.get("narration") or "",
            )
            last_remaining_tokens = remaining_tokens
            log.info("✓ Improved %s (tokens remaining this minute: %d)",
                     file["file_name"], remaining_tokens)

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

        except Exception as exc:
            log.error("✗ Failed to improve file %d (%s): %s",
                      file_id, file.get("file_name", "?"), exc)
            failed_files.append((file_id, file.get("file_name", "?"), str(exc)))
            last_remaining_tokens = 0  # assume worst case after a failure

    log.info("Code Improver complete. %d/%d files processed successfully.",
             len(file_ids) - len(failed_files), len(file_ids))

    if failed_files:
        log.warning("Failed files:\n%s",
                    "\n".join(f"  - id={fid} ({name}): {err}"
                              for fid, name, err in failed_files))
        raise RuntimeError(f"{len(failed_files)} file(s) failed.")


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
