# airflow/dags/life_os_code_narrate.py
"""
Code Narrator DAG — triggered from LifeOS Code Intelligence UI.
Conf required: {"file_ids": [1, 2, 3], "project_id": 42}

Changes vs original: same throttling and retry improvements as
life_os_code_improve.py — see that file for the full rationale.

Note: The Narrator was working before because it was typically run on
fewer files at a time. Adding the delay here anyway for consistency
and to prevent issues when narrating large projects.
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

INTER_REQUEST_DELAY_SEC = 3

default_args = {
    "owner":            "life_os",
    "retries":          2,
    "retry_delay":      timedelta(minutes=3),
    "email_on_failure": False,
}


def task_narrate_files(**context):
    from dag_db import fetch_one, execute

    conf     = context["dag_run"].conf or {}
    file_ids = conf.get("file_ids", [])
    if not file_ids:
        raise ValueError("file_ids list is required in conf")

    log.info(
        "Code Narrator starting. %d file(s) to process. "
        "Inter-request delay: %ds.",
        len(file_ids), INTER_REQUEST_DELAY_SEC,
    )

    failed_files = []

    for idx, file_id in enumerate(file_ids):
        if idx > 0:
            log.debug("Waiting %ds before next request (%d/%d)...",
                      INTER_REQUEST_DELAY_SEC, idx + 1, len(file_ids))
            time.sleep(INTER_REQUEST_DELAY_SEC)

        file = fetch_one("SELECT * FROM code_files WHERE id = %s", (file_id,))
        if not file:
            log.warning("File %d not found, skipping", file_id)
            continue
        if not file.get("raw_code"):
            log.warning("File %d (%s) has no raw_code — pull it first, skipping",
                        file_id, file.get("file_name"))
            continue

        # Get project context for better narration quality
        project = fetch_one(
            "SELECT * FROM code_projects WHERE id = %s", (file["project_id"],)
        )
        readme_context = (
            (project.get("readme_md") or project.get("description") or "")
            if project else ""
        )

        log.info("Narrating %s (%d/%d)", file["file_name"], idx + 1, len(file_ids))
        try:
            from agents.blog_agents import agent_code_narrator
            narration = agent_code_narrator(
                code_content=file["raw_code"],
                file_name=file["file_name"],
                readme_context=readme_context,
            )
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            execute(
                """UPDATE code_files
                   SET narration = %s,
                       narration_generated_at = %s,
                       updated_at = %s
                   WHERE id = %s""",
                (narration, now, now, file_id),
            )
            log.info("✓ Narrated %s", file["file_name"])

        except Exception as exc:
            log.error(
                "✗ Failed to narrate file %d (%s) after retries: %s",
                file_id, file.get("file_name", "?"), exc,
            )
            failed_files.append((file_id, file.get("file_name", "?"), str(exc)))

    log.info(
        "Code Narrator complete. %d/%d files processed successfully.",
        len(file_ids) - len(failed_files), len(file_ids),
    )

    if failed_files:
        log.warning(
            "Failed files (can be re-triggered individually):\n%s",
            "\n".join(f"  - id={fid} ({name}): {err}" for fid, name, err in failed_files),
        )
        raise RuntimeError(
            f"{len(failed_files)} file(s) failed. Check logs above for details."
        )


with DAG(
    dag_id="life_os_code_narrate",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "code_intel"],
) as dag:
    PythonOperator(
        task_id="narrate_files",
        python_callable=task_narrate_files,
    )
