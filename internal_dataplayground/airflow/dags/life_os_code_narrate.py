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


def task_narrate_files(**context):
    from dag_db import fetch_one, execute
    from agents.blog_agents import _cerebras, _wait_if_needed, _CEREBRAS_QWEN3
    from agents.blog_agents import _detect_file_type, _NARRATOR_FOCUS

    conf       = context["dag_run"].conf or {}
    file_ids   = conf.get("file_ids", [])
    project_id = conf.get("project_id")
    if not file_ids:
        raise ValueError("file_ids required in DAG conf")

    log.info("Code Narrator: %d file(s), header-driven rate limiting", len(file_ids))

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

        project = fetch_one(
            "SELECT * FROM code_projects WHERE id = %s",
            (file.get("project_id") or project_id,)
        )
        readme_context = (
            (project.get("readme_md") or project.get("description") or "")
            if project else ""
        )

        log.info("Narrating %s (%d/%d)", file["file_name"], i + 1, len(file_ids))

        try:
            file_type  = _detect_file_type(file["file_name"])
            type_focus = _NARRATOR_FOCUS.get(file_type, _NARRATOR_FOCUS["other"])

            system = f"""
You are a Principal Engineer reviewing a file for two audiences:
  1. Other AI agents that need structured intelligence about this file.
  2. A hiring manager reading a portfolio project.

Your output is a technical narration — NOT a blog post.
Write as a dense, analytical briefing for a senior engineer.
Do NOT explain basic syntax or standard library usage.

{type_focus}

MANDATORY EDITORIAL LENSES — always include both:

  🟢 THE MOST INTERESTING THING
     What is elegant, clever, or non-obvious here?
     What decision or pattern would be worth writing a blog post about?
     Be specific — name the function, pattern, or line range.

  🔴 THE MOST FRAGILE THING
     What is the highest-risk thing for a developer who doesn't know this codebase?
     What could break silently, what assumption is buried, what would a new
     contributor get wrong?
     Be specific and honest.

FORMAT: Markdown, short headers, under 700 words.
Prioritise signal — a sharp 400-word narration beats an exhaustive 700-word one.
"""
            project_ctx = f"Project context: {readme_context}\n" if readme_context else ""
            prompt = f"File: {file['file_name']}\n{project_ctx}Code:\n{file['raw_code']}"

            narration, last_rate_limits = _cerebras(
                _CEREBRAS_QWEN3, system, prompt, temperature=0.25
            )

            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            execute(
                """UPDATE code_files
                   SET narration = %s, narration_generated_at = %s, updated_at = %s
                   WHERE id = %s""",
                (narration, now, now, file_id),
            )
            log.info("✓ Narrated %s", file["file_name"])
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
    dag_id="life_os_code_narrate",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "code_intel"],
) as dag:
    PythonOperator(task_id="narrate_files", python_callable=task_narrate_files)