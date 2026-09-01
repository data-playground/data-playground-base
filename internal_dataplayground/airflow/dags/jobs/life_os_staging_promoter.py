# airflow/dags/life_os_staging_promoter.py
"""
Staging Promoter DAG
─────────────────────
Companion to life_os_job_scout.py — promotes manually-queued StagingJob rows
(added via the "Add Job Link" panel on /jobs, POST /jobs/stage) into
linkedin_jobs, the same table both Job Scout DAGs write to and the same
table the Jobs page reads from.

Mirrors life_os_job_scout.py's pipeline shape (scrape detail -> score ->
load -> log run) but SKIPS step 1 (LinkedIn search) entirely — a staged
job already has its job_link, so this jumps straight to step 2: fetch the
detail page, this time via job_agents.get_full_job_posting() instead of
get_job_details(), since a staged job has no search-result-card data to
seed job_title/company_name/location from the way a scheduled search does.
See that function's docstring for an important caveat about its
title/company/location selectors being unverified against a live page.

TRIGGER MODES:
  - Scheduled: runs once daily, catching anything left PENDING (e.g. the
    FastAPI container was down when a job was queued, or a previous run's
    scrape/score step failed and the row needs a retry).
  - Manual / on-demand: POST /jobs/stage/process (see
    domains/jobs/routers/staging.py) triggers this DAG immediately via the
    "⚡ Process Now" button next to the staging queue, so a freshly queued
    job gets processed within seconds instead of waiting for the next
    scheduled run. That endpoint triggers with no conf (see below) — it
    just asks the DAG to run, and the DAG itself figures out what's
    PENDING at run time.

Conf (optional, for backfill/troubleshooting a specific row):
  {"staging_ids": [12, 13]}   # limits the run to these staging_jobs.id values

ARCHITECTURAL RULE (see CONTRIBUTING.md): this DAG never imports models.py,
database.py, or any router/service. All DB access goes through dag_db.py.
"""
import logging
import time
from datetime import datetime, timedelta

import sys

sys.path.insert(0, '/opt/airflow/project')
sys.path.insert(0, '/opt/airflow/project/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

default_args = {
    "owner": "life_os",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


def task_fetch_and_scrape(**context):
    from dag_db import fetch_all, execute_many
    from agents.job_agents import get_full_job_posting, extract_linkedin_job_id, DETAIL_FETCH_DELAY_SEC

    conf = context["dag_run"].conf or {}
    staging_ids = conf.get("staging_ids")

    sql = "SELECT id, job_link, job_search FROM staging_jobs WHERE status = 'PENDING'"
    params = None
    if staging_ids:
        placeholders = ", ".join(["%s"] * len(staging_ids))
        sql += f" AND id IN ({placeholders})"
        params = tuple(staging_ids)

    pending = fetch_all(sql, params) if params else fetch_all(sql)
    context["ti"].xcom_push(key="items_attempted", value=len(pending))

    if not pending:
        log.info("No PENDING staging_jobs rows to process — skipping")
        context["ti"].xcom_push(key="scraped", value=[])
        return

    # Mark all of them PROCESSING up front, before any scraping starts, so a
    # second trigger (e.g. someone double-clicking "Process Now") can't pick
    # up and double-process the same rows mid-run.
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    execute_many([
        (
            "UPDATE staging_jobs SET status = 'PROCESSING', updated_at = %s WHERE id = %s",
            (now_str, row["id"]),
        )
        for row in pending
    ])

    scraped = []
    for idx, row in enumerate(pending):
        if idx > 0:
            time.sleep(DETAIL_FETCH_DELAY_SEC)

        posting = get_full_job_posting(row["job_link"])

        if not posting.get("description"):
            # Total fetch failure (or the page loaded but the markup didn't
            # match) — fail this row now rather than letting it silently
            # fall through the scoring/load steps with nothing to score.
            execute_many([(
                "UPDATE staging_jobs SET status = 'FAILED', error_message = %s, updated_at = %s "
                "WHERE id = %s",
                ("Could not fetch or parse the job posting page.", now_str, row["id"]),
            )])
            log.warning("Staging job %s: fetch/parse failed, marked FAILED", row["id"])
            continue

        linkedin_job_id = extract_linkedin_job_id(row["job_link"])
        scraped.append({
            "staging_id":   row["id"],
            "job_link":     row["job_link"],
            # Falls back to a value unique per staging row when the URL
            # doesn't match the expected pattern, so every job still has
            # a distinct identifier to score/dedupe/insert with.
            "job_id":       linkedin_job_id or f"staging-{row['id']}",
            "job_title":    posting["job_title"],
            "company_name": posting["company_name"],
            "location":     posting["location"],
            "description":  posting["description"],
            "salary":       posting["salary"],
            "job_search":   row["job_search"] or "staged",
        })

    context["ti"].xcom_push(key="scraped", value=scraped)
    log.info("Scrape complete: %d/%d staged jobs ready for scoring", len(scraped), len(pending))


def task_score(**context):
    from agents.job_agents import build_scoring_chunks, score_job_batch
    from agents.job_resume_context import RESUME_MARKDOWN, KEY_STRENGTHS_TO_WEIGHT

    jobs = context["ti"].xcom_pull(key="scraped", task_ids="fetch_and_scrape") or []
    if not jobs:
        log.info("No scraped staging jobs to score — skipping")
        context["ti"].xcom_push(key="enriched", value=[])
        return

    chunks = build_scoring_chunks(jobs)
    log.info("Scoring %d staged jobs across %d batches", len(jobs), len(chunks))

    results_by_id: dict[str, dict] = {}
    for i, chunk in enumerate(chunks):
        try:
            results = score_job_batch(chunk, RESUME_MARKDOWN, KEY_STRENGTHS_TO_WEIGHT)
            for r in results:
                results_by_id[r["ID"]] = r
        except Exception as exc:
            log.error("Batch %d failed to score — %d staged jobs skipped: %s", i + 1, len(chunk), exc)
        if i < len(chunks) - 1:
            time.sleep(2)

    enriched = []
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for job in jobs:
        analysis = results_by_id.get(job["job_id"])
        if not analysis:
            from dag_db import execute_many
            execute_many([(
                "UPDATE staging_jobs SET status = 'FAILED', error_message = %s, updated_at = %s "
                "WHERE id = %s",
                ("AI fit-scoring did not return a result for this job.", now_str, job["staging_id"]),
            )])
            log.warning("Staging job %s: no AI analysis returned, marked FAILED", job["staging_id"])
            continue
        enriched.append({**job, **{
            "remote":                 analysis["remote"],
            "explanation":            analysis["explanation"],
            "fit_score":              analysis["fit_score"],
            "qualification_analysis": analysis["qualification_analysis"],
            "skill_gaps":             analysis["skill_gaps"],
        }})

    context["ti"].xcom_push(key="enriched", value=enriched)
    log.info("Scoring complete: %d staged jobs enriched and ready to load", len(enriched))


def task_load(**context):
    from dag_db import fetch_all, execute_many

    enriched = context["ti"].xcom_pull(key="enriched", task_ids="score") or []
    if not enriched:
        log.info("Nothing to insert this run.")
        return

    # Dedup safety net: a staged job might be the exact same posting the
    # scheduled Job Scout DAG already found via search (or a previous,
    # since-failed run of this same DAG partially inserted it). Skip
    # inserting it again — the (source, external_ref) UNIQUE constraint on
    # Job would reject it anyway, but checking first lets us still mark the
    # staging row DONE (with a note) instead of leaving it stuck.
    existing_rows = fetch_all(
        "SELECT job_id, external_ref FROM linkedin_jobs WHERE source = 'linkedin'"
    )
    existing_job_ids = {str(r["job_id"]) for r in existing_rows if r["job_id"] is not None}
    existing_external_refs = {r["external_ref"] for r in existing_rows if r["external_ref"] is not None}

    columns = [
        "job_id", "source", "external_ref", "job_title", "company_name", "job_link",
        "salary", "remote", "location", "post_date", "description", "explanation",
        "qualification_analysis", "skill_gaps", "fit_score", "job_search", "search_date",
    ]
    placeholders = ", ".join(["%s"] * len(columns))
    insert_sql = f"INSERT INTO linkedin_jobs ({', '.join(columns)}) VALUES ({placeholders})"

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    statements = []
    # Mixed UPDATE statements against BOTH linkedin_jobs (search_date
    # refresh, only for the already-present branch) and staging_jobs
    # (status -> DONE, every job) — execute_many doesn't care which table
    # each statement targets, so these run together in one batch.
    db_updates = []
    loaded = 0

    for job in enriched:
        linkedin_id = job["job_id"] if job["job_id"] and not job["job_id"].startswith("staging-") else None
        already_present = (
            (linkedin_id and linkedin_id in existing_job_ids)
            or job["job_id"] in existing_external_refs
        )

        if already_present:
            log.info(
                "Staging job %s: posting already in linkedin_jobs (job_id/external_ref match) — "
                "skipping insert, refreshing search_date, marking DONE", job["staging_id"],
            )
            # Same "still open" signal as the two scheduled scout DAGs —
            # a manually-pasted link matching an existing row means this
            # posting was just confirmed to still exist today too.
            db_updates.append((
                "UPDATE linkedin_jobs SET search_date = %s WHERE job_id = %s OR external_ref = %s",
                (today_str, linkedin_id, job["job_id"]),
            ))
        else:
            row = (
                int(linkedin_id) if linkedin_id else None,
                "linkedin",
                job["job_id"],  # numeric linkedin id as string, or "staging-<id>" fallback
                job.get("job_title"),
                job.get("company_name"),
                job.get("job_link"),
                job.get("salary"),
                1 if job.get("remote") else 0,
                job.get("location"),
                None,  # post_date — not reliably available from the detail page alone
                job.get("description"),
                job.get("explanation"),
                job.get("qualification_analysis"),
                job.get("skill_gaps"),
                job.get("fit_score"),
                job.get("job_search"),
                today_str,
            )
            statements.append((insert_sql, row))
            loaded += 1

        db_updates.append((
            "UPDATE staging_jobs SET status = 'DONE', job_id = %s, job_title = %s, "
            "company_name = %s, location = %s, salary = %s, description = %s, updated_at = %s "
            "WHERE id = %s",
            (
                linkedin_id, job.get("job_title"), job.get("company_name"),
                job.get("location"), job.get("salary"), job.get("description"),
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), job["staging_id"],
            ),
        ))

    if statements:
        execute_many(statements)
    execute_many(db_updates)

    context["ti"].xcom_push(key="items_loaded", value=loaded)
    log.info("Staging promotion complete: %d/%d staged jobs inserted into linkedin_jobs", loaded, len(enriched))


def task_log_run(**context):
    """
    Writes this run's outcome to job_scout_run_log regardless of whether
    upstream tasks succeeded (trigger_rule="all_done" below), same pattern
    as both existing Job Scout DAGs use for their own health logging.
    """
    from agents.job_scout_health import log_run

    items_attempted = context["ti"].xcom_pull(key="items_attempted", task_ids="fetch_and_scrape") or 0
    scraped = context["ti"].xcom_pull(key="scraped", task_ids="fetch_and_scrape") or []
    enriched = context["ti"].xcom_pull(key="enriched", task_ids="score") or []
    items_loaded = context["ti"].xcom_pull(key="items_loaded", task_ids="load") or 0

    try:
        log_run(
            dag_id="life_os_staging_promoter",
            items_attempted=items_attempted,
            items_found=len(scraped),
            new_items=len(enriched),
            items_loaded=items_loaded,
        )
    except Exception as exc:
        log.error("Failed to write job_scout_run_log: %s", exc)


with DAG(
    dag_id="life_os_staging_promoter",
    default_args=default_args,
    schedule_interval="0 6 * * *",   # once daily — catches anything left PENDING
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "jobs"],
    doc_md="""
## Staging Promoter DAG

Processes PENDING rows in `staging_jobs` (queued via the "Add Job Link"
panel on /jobs) straight through to `linkedin_jobs` — fetch detail page,
score fit via Gemini, insert, mark the staging row DONE/FAILED. Same
destination table as both Job Scout DAGs; the Jobs page needs no changes
to pick these up.

**Manual trigger, limited to specific staging rows:**
```json
{"staging_ids": [12, 13]}
```

**Manual trigger, process everything currently PENDING** (what the
"⚡ Process Now" button on /jobs sends): trigger with no conf at all.
    """,
) as dag:

    fetch = PythonOperator(task_id="fetch_and_scrape", python_callable=task_fetch_and_scrape)
    score = PythonOperator(task_id="score", python_callable=task_score)
    load  = PythonOperator(task_id="load", python_callable=task_load)
    log_run_task = PythonOperator(
        task_id="log_run", python_callable=task_log_run, trigger_rule="all_done",
    )

    fetch >> score >> load >> log_run_task
