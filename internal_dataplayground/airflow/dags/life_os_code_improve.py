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

default_args = {
    "owner":             "life_os",
    "retries":           2,           # was 0 — allows recovery from transient 429 windows
    "retry_delay":       timedelta(minutes=3),
    "email_on_failure":  False,
}


def task_improve_files(**context):
    from dag_db import fetch_one, execute
    from agents.blog_agents import (
        _cerebras, _wait_if_needed, _CEREBRAS_QWEN3,
        LARGE_FILE_THRESHOLD_TOKENS, _estimate_tokens,
    )

    file_ids = (context["dag_run"].conf or {}).get("file_ids", [])
    if not file_ids:
        raise ValueError("file_ids required in DAG conf")

    log.info("Code Improver: %d file(s), header-driven rate limiting", len(file_ids))

    # Pre-flight: warn on large files so the operator knows in advance
    for file_id in file_ids:
        f = fetch_one(
            "SELECT file_name, LENGTH(raw_code) AS char_len "
            "FROM code_files WHERE id = %s", (file_id,)
        )
        if f and f.get("char_len") and (f["char_len"] // 4) > LARGE_FILE_THRESHOLD_TOKENS:
            log.warning(
                "Large file: %s (~%d tokens) — File Structure Recommendation "
                "will be included in the report.",
                f["file_name"], f["char_len"] // 4,
            )
 
    succeeded       = 0
    failed          = 0
    last_rate_limits = {}
 
    for i, file_id in enumerate(file_ids):
        file = fetch_one("SELECT * FROM code_files WHERE id = %s", (file_id,))
        if not file or not file.get("raw_code"):
            log.warning("File %d not found or no raw_code — skipping", file_id)
            failed += 1
            continue
 
        # Gate on headers from the previous call — skip for first iteration
        if i > 0 and last_rate_limits:
            _wait_if_needed(last_rate_limits, file["file_name"])
 
        log.info("Improving %s (%d/%d)", file["file_name"], i + 1, len(file_ids))
 
        try:
            token_estimate = _estimate_tokens(file["raw_code"])
            is_large       = token_estimate > LARGE_FILE_THRESHOLD_TOKENS
            narration      = file.get("narration") or ""
 
            large_file_instruction = ""
            if is_large:
                large_file_instruction = f"""
⚠ LARGE FILE (~{token_estimate:,} tokens). You MUST include a final section:
 
## File Structure Recommendation
**Category:** Maintainability  **Severity:** Medium
Frame as a genuine software quality concern. Name each proposed new module,
its single responsibility, and which functions/classes move there.
"""
 
            system = f"""
You are a Principal Engineer doing a thorough code review.
Produce an improvement report — do NOT rewrite the code.
 
FORMAT per suggestion:
## [Short specific title]
**Lines:** [range or function]  **Category:** Performance|Readability|Correctness|Security|Maintainability|Testing  **Severity:** Low|Medium|High
 
*What's happening:* [1-2 sentences — formal, precise, scannable]
 
*Why it matters here:* [1-3 sentences — conversational PR-comment tone,
reference the specific codebase context and what breaks in production]
 
*Suggestion:*
```[language]
# Smallest change that fixes the issue
```
 
ALWAYS CHECK FOR:
  SQLAlchemy: N+1 risks, missing await, wrong lazy strategy, commit-before-refresh
  FastAPI: endpoints doing too much, missing HTTPException, response model mismatches
  Airflow: DB logic in DAG files, missing retries on external calls, non-idempotent tasks
  General: hardcoded strings, single-responsibility violations, silent exception swallowing
  Testing: flag untested high-risk functions with the specific failure mode a test would catch
 
ORDER: High severity first. MAX: 10 suggestions (plus File Structure if applicable).
{large_file_instruction}"""
 
            context_block = f"Context:\n{narration[:600]}\n\n" if narration else ""
            prompt = f"File: {file['file_name']}\n{context_block}Code:\n{file['raw_code']}"
 
            notes, last_rate_limits = _cerebras(
                _CEREBRAS_QWEN3, system, prompt, temperature=0.2
            )
 
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            execute(
                """UPDATE code_files
                   SET improvement_notes = %s, improvement_generated_at = %s,
                       improvement_status = %s, updated_at = %s
                   WHERE id = %s""",
                (notes, now, "generated", now, file_id),
            )
            log.info("✓ Improved %s", file["file_name"])
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
    dag_id="life_os_code_improve",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "code_intel"],
) as dag:
    PythonOperator(task_id="improve_files", python_callable=task_improve_files)
 