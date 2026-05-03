# airflow/dags/life_os_code_comment.py
"""
Code Commenter DAG — triggered from LifeOS Code Intelligence UI.
Conf required: {"file_ids": [1, 2, 3]}

Changes vs original: same throttling and retry improvements as
life_os_code_improve.py — see that file for the full rationale.
"""

import sys
import logging
from datetime import datetime, timedelta

sys.path.insert(0, '/opt/airflow/project')
sys.path.insert(0, '/opt/airflow/project/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

default_args = {
    "owner":            "life_os",
    "retries":          2,
    "retry_delay":      timedelta(minutes=3),
    "email_on_failure": False,
}


def task_comment_files(**context):
    from dag_db import fetch_one, execute
    from agents.blog_agents import (
        _cerebras, _wait_if_needed, _CEREBRAS_QWEN3,
        _detect_file_type, _COMMENTER_CONVENTIONS,
    )

    file_ids = (context["dag_run"].conf or {}).get("file_ids", [])
    if not file_ids:
        raise ValueError("file_ids required in DAG conf")

    log.info("Code Commenter: %d file(s), header-driven rate limiting", len(file_ids))

    succeeded        = 0
    failed           = 0
    last_rate_limits = {}

    for i, file_id in enumerate(file_ids):
        file = fetch_one("SELECT * FROM code_files WHERE id = %s", (file_id,))
        if not file or not file.get("raw_code"):
            log.warning("File %d not found or no raw_code — skipping", file_id)
            failed += 1
            continue

        if i > 0 and last_rate_limits:
            _wait_if_needed(last_rate_limits, file["file_name"])

        log.info("Commenting %s (%d/%d)", file["file_name"], i + 1, len(file_ids))

        try:
            file_type   = _detect_file_type(file["file_name"])
            conventions = _COMMENTER_CONVENTIONS.get(
                file_type, _COMMENTER_CONVENTIONS["other"]
            )

            system = f"""
You are a Senior Engineer performing a documentation pass on a codebase.
Your ONLY job is to add comments and docstrings. You MUST NOT change any
logic, rename anything, reorder code, or refactor anything whatsoever.

FILE-TYPE CONVENTIONS:
{conventions}

COMMENTING PHILOSOPHY:

  GROUP OVER LINE:
    A single section divider above related lines is better than
    individual inline comments on each line. Ask: do these lines share
    a single purpose? If yes, one comment above the group is enough.

  CALIBRATE "OBVIOUS" CORRECTLY:
    Obvious means obvious to a senior developer who knows this language
    AND the tools in this file (FastAPI, SQLAlchemy async, Airflow, HTMX).
    Do not comment session.commit(), standard ORM columns, standard HTML,
    or CSS properties whose purpose is clear from the name.

  WHAT ALWAYS GETS COMMENTED:
    - Non-obvious "why" decisions (why this loading strategy, why this status code)
    - Workarounds and their reason
    - Logic that could silently break if changed
    - Group boundaries where the next block does something meaningfully different

OUTPUT: The COMPLETE file with comments added.
No preamble. No explanation. Just the file content, starting from line 1.
"""
            prompt = f"Add comments to {file['file_name']}:\n\n{file['raw_code']}"

            commented, last_rate_limits = _cerebras(
                _CEREBRAS_QWEN3, system, prompt, temperature=0.2
            )

            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            execute(
                """UPDATE code_files
                   SET commented_code = %s, commented_generated_at = %s,
                       commented_status = %s, updated_at = %s
                   WHERE id = %s""",
                (commented, now, "generated", now, file_id),
            )
            log.info("✓ Commented %s", file["file_name"])
            succeeded += 1

        except RuntimeError as exc:
            if "daily request limit" in str(exc):
                log.error("Daily limit exhausted — aborting")
                break
            log.error("Failed %d (%s): %s", file_id, file["file_name"], exc)
            failed += 1
        except Exception as exc:
            log.error("Failed %d (%s): %s", file_id, file["file_name"], exc)
            failed += 1

    log.info("Complete: %d succeeded, %d failed of %d", succeeded, failed, len(file_ids))
    if failed:
        raise RuntimeError(f"{failed} file(s) failed — see logs above")


with DAG(
    dag_id="life_os_code_comment",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "code_intel"],
) as dag:
    PythonOperator(task_id="comment_files", python_callable=task_comment_files)