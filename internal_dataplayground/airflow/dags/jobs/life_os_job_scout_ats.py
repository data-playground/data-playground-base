# airflow/dags/life_os_job_scout_ats.py
"""
Job Scout — ATS DAG
────────────────────
Companion to life_os_job_scout.py. Reads watched_companies (the curated
Greenhouse/Lever list managed at /jobs/config), pulls every open posting
directly from each company's public job board API, scores fit via the same
Gemini pipeline, and inserts enriched rows into linkedin_jobs — the same
table both this DAG and the LinkedIn scraper write to, and the same table
the Jobs page reads from.

No HTML scraping here — Greenhouse/Lever expose clean JSON with full
descriptions inline, so there's no separate "detail page" fetch step and
none of the rate-limit/blocking risk that comes with LinkedIn.

Conf (optional, for manual triggers):
  {"companies": ["Exact Company Name", "..."]}   # limits the run to these
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
    "owner": "life_os",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


def task_fetch_ats_jobs(**context):
    from dag_db import fetch_all, execute_many
    from agents.job_ats_agents import fetch_all_watched_companies

    conf = context["dag_run"].conf or {}
    name_filter = set(conf.get("companies") or [])

    companies = fetch_all(
        "SELECT company_name, greenhouse_slug, lever_slug FROM watched_companies "
        "WHERE is_active = 1 AND (greenhouse_slug IS NOT NULL OR lever_slug IS NOT NULL)"
    )
    if name_filter:
        companies = [c for c in companies if c["company_name"] in name_filter]

    if not companies:
        log.info("No active watched companies with a Greenhouse/Lever slug — nothing to fetch")
        context["ti"].xcom_push(key="ats_jobs", value=[])
        return

    log.info("Fetching postings for %d watched companies", len(companies))
    all_jobs = fetch_all_watched_companies(companies)

    existing = fetch_all(
        "SELECT source, external_ref FROM linkedin_jobs WHERE source IN ('greenhouse', 'lever')"
    )
    existing_keys = {(row["source"], row["external_ref"]) for row in existing}

    new_jobs = [j for j in all_jobs if (j["source"], j["external_ref"]) not in existing_keys]
    rescanned = [j for j in all_jobs if (j["source"], j["external_ref"]) in existing_keys]
    log.info(
        "%d postings found across watched companies, %d are new, %d are re-scans of existing rows",
        len(all_jobs), len(new_jobs), len(rescanned),
    )

    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    # Postings still open today — refresh search_date instead of leaving it
    # frozen at first-seen. One UPDATE per (source, external_ref) since
    # there's no single-column key to batch on the way the LinkedIn DAG's
    # job_id lets it (external_ref alone isn't unique across sources).
    if rescanned:
        execute_many([
            (
                "UPDATE linkedin_jobs SET search_date = %s WHERE source = %s AND external_ref = %s",
                (today_str, j["source"], j["external_ref"]),
            )
            for j in rescanned
        ])
        log.info("Refreshed search_date for %d re-scanned ATS postings", len(rescanned))

    # Cross-source dedup: skip anything that's the same role already sitting
    # in linkedin_jobs from the LinkedIn DAG (same company + near-identical title).
    from agents.job_dedup import filter_cross_source_duplicates, RECENCY_WINDOW_DAYS
    cutoff = (datetime.utcnow() - timedelta(days=RECENCY_WINDOW_DAYS)).strftime("%Y-%m-%d")
    recent_rows = fetch_all(
        "SELECT company_name, job_title, source FROM linkedin_jobs WHERE search_date >= %s",
        (cutoff,),
    )
    new_jobs, skipped_dupes = filter_cross_source_duplicates(new_jobs, recent_rows)
    if skipped_dupes:
        log.info("Skipped %d likely cross-source duplicates (already on LinkedIn)", skipped_dupes)

    context["ti"].xcom_push(key="items_attempted", value=len(companies))
    context["ti"].xcom_push(key="items_found", value=len(all_jobs))

    for job in new_jobs:
        job["search_date"] = today_str

    context["ti"].xcom_push(key="ats_jobs", value=new_jobs)


def task_score_jobs(**context):
    from agents.job_agents import build_scoring_chunks, score_job_batch
    from agents.job_resume_context import RESUME_MARKDOWN, KEY_STRENGTHS_TO_WEIGHT

    jobs = context["ti"].xcom_pull(key="ats_jobs", task_ids="fetch_ats_jobs") or []
    if not jobs:
        log.info("No new ATS jobs to score — skipping")
        context["ti"].xcom_push(key="enriched_jobs", value=[])
        return

    chunks = build_scoring_chunks(jobs)
    log.info("Scoring %d ATS jobs across %d batches", len(jobs), len(chunks))

    results_by_ref: dict[str, dict] = {}
    for i, chunk in enumerate(chunks):
        try:
            results = score_job_batch(chunk, RESUME_MARKDOWN, KEY_STRENGTHS_TO_WEIGHT)
            for r in results:
                results_by_ref[r["ID"]] = r
        except Exception as exc:
            log.error("Batch %d failed to score — %d jobs skipped: %s", i + 1, len(chunk), exc)
        if i < len(chunks) - 1:
            time.sleep(2)

    enriched = []
    for job in jobs:
        analysis = results_by_ref.get(job["external_ref"])
        if not analysis:
            log.warning("No AI analysis for %s/%s (%s) — skipping insert",
                        job["source"], job["external_ref"], job["job_title"])
            continue
        enriched.append({
            **job,
            "remote":                 analysis["remote"],
            "explanation":            analysis["explanation"],
            "fit_score":              analysis["fit_score"],
            "qualification_analysis": analysis["qualification_analysis"],
            "skill_gaps":             analysis["skill_gaps"],
        })

    context["ti"].xcom_push(key="enriched_jobs", value=enriched)
    log.info("Scoring complete: %d ATS jobs enriched", len(enriched))


def task_load_to_mysql(**context):
    from dag_db import execute_many

    enriched = context["ti"].xcom_pull(key="enriched_jobs", task_ids="score_jobs") or []
    if not enriched:
        log.info("Nothing to insert this run.")
        return

    columns = [
        "job_id", "source", "external_ref", "job_title", "company_name", "job_link",
        "salary", "remote", "location", "post_date", "description", "explanation",
        "qualification_analysis", "skill_gaps", "fit_score", "job_search", "search_date",
    ]
    placeholders = ", ".join(["%s"] * len(columns))
    sql = f"INSERT INTO linkedin_jobs ({', '.join(columns)}) VALUES ({placeholders})"

    statements = []
    for job in enriched:
        row = (
            None,  # job_id — legacy numeric column doesn't apply to ATS sources
            job["source"],
            job["external_ref"],
            job.get("job_title"),
            job.get("company_name"),
            job.get("job_link"),
            job.get("salary"),
            1 if job.get("remote") else 0,
            job.get("location"),
            job.get("post_date"),
            job.get("description"),
            job.get("explanation"),
            job.get("qualification_analysis"),
            job.get("skill_gaps"),
            job.get("fit_score"),
            job.get("job_search"),
            job.get("search_date"),
        )
        statements.append((sql, row))

    execute_many(statements)
    log.info("Inserted %d ATS jobs into linkedin_jobs", len(statements))


def task_log_run(**context):
    from agents.job_scout_health import log_run

    items_attempted = context["ti"].xcom_pull(key="items_attempted", task_ids="fetch_ats_jobs") or 0
    items_found = context["ti"].xcom_pull(key="items_found", task_ids="fetch_ats_jobs") or 0
    fetched = context["ti"].xcom_pull(key="ats_jobs", task_ids="fetch_ats_jobs") or []
    enriched = context["ti"].xcom_pull(key="enriched_jobs", task_ids="score_jobs") or []

    try:
        log_run(
            dag_id="life_os_job_scout_ats",
            items_attempted=items_attempted,
            items_found=items_found,
            new_items=len(fetched),
            items_loaded=len(enriched),
        )
    except Exception as exc:
        log.error("Failed to write job_scout_run_log: %s", exc)


with DAG(
    dag_id="life_os_job_scout_ats",
    default_args=default_args,
    schedule_interval="0 */6 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "jobs"],
    doc_md="""
## Job Scout — ATS DAG

Reads `watched_companies` (managed at /jobs/config), pulls open postings
directly from each company's Greenhouse/Lever board, scores fit via Gemini,
and inserts into `linkedin_jobs` alongside the LinkedIn-sourced rows.

**Manual trigger limited to specific companies:**
```json
{"companies": ["Anthropic", "Figma"]}
```
    """,
) as dag:

    fetch = PythonOperator(task_id="fetch_ats_jobs", python_callable=task_fetch_ats_jobs)
    score = PythonOperator(task_id="score_jobs",      python_callable=task_score_jobs)
    load  = PythonOperator(task_id="load_to_mysql",   python_callable=task_load_to_mysql)
    log_run_task = PythonOperator(
        task_id="log_run", python_callable=task_log_run, trigger_rule="all_done",
    )

    fetch >> score >> load >> log_run_task
