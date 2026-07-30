# airflow/dags/life_os_job_scout.py
"""
Job Scout DAG
─────────────
Replaces the standalone LinkedInJobScraper script. Runs on a schedule,
searches LinkedIn for the configured job titles, skips job_ids already in
linkedin_jobs, pulls each job's full description, scores every job's fit
against the resume via Gemini, and inserts the enriched rows into
linkedin_jobs — the same table the Jobs page in the FastAPI app already
reads from. No frontend or router changes are required for jobs to start
showing up again.

ARCHITECTURAL RULE (see CONTRIBUTING.md): this DAG never imports models.py,
database.py, or any router/service. All DB access goes through dag_db.py.

Conf (optional, for manual/backfill triggers):
  {"searches": ["Custom Search Title", "Another One"]}   # overrides DEFAULT_SEARCHES

Tuning notes:
  - DETAIL_FETCH_DELAY_SEC in job_agents.py controls the pause between
    per-job detail page fetches — the first knob to turn if LinkedIn starts
    blocking or serving empty descriptions.
  - Gemini calls run through job_agents.score_job_batch, which retries on
    429/503 the same way the blog pipeline agents do.
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

# Same search list as the original script. Add/remove queries here — no
# code changes needed elsewhere. Can also be overridden per-run via conf.
DEFAULT_SEARCHES = [
    "Senior Analytics Engineer", "AI Solutions Architect", "Senior BI Engineer",
    "Senior Data Analyst", "Senior Data Scientist", "Senior Data Engineer",
    "Senior Machine Learning Engineer", "Senior AI Engineer", "Senior Analytics Manager",
    "Data Engineer (GCP), Solutions Engineer (Vertex AI), Analytics Architect",
    "Revenue Operations Engineer", "Product Data Scientist",
    "Senior Data Engineer AND (GCP OR BigQuery) AND Full-time",
    "(AI Architect OR Analytics Engineer) AND (Vertex OR GenAI) -Contract",
    "(Analytics Engineer OR Data Engineer) AND (Python AND SQL) AND Senior",
]

default_args = {
    "owner": "life_os",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


def task_search_and_scrape(**context):
    from dag_db import fetch_all
    from agents.job_agents import (
        search_linkedin_jobs, get_job_details, deduplicate_jobs,
        clean_date, DETAIL_FETCH_DELAY_SEC,
    )

    conf = context["dag_run"].conf or {}

    if conf.get("searches"):
        # Manual trigger override — doesn't touch job_search_keywords.
        searches = conf["searches"]
    else:
        # Normal path: read the editable list from the DB (managed via the
        # Jobs > Config page / routers/job_config.py). Falls back to the
        # hardcoded DEFAULT_SEARCHES only if the table is somehow empty —
        # this should only happen before the seed migration has run.
        rows = fetch_all("SELECT keyword FROM job_search_keywords WHERE is_active = 1")
        searches = [row["keyword"] for row in rows] or DEFAULT_SEARCHES
        if not rows:
            log.warning("job_search_keywords returned no active rows — falling back to DEFAULT_SEARCHES")

    existing_rows = fetch_all("SELECT DISTINCT job_id FROM linkedin_jobs")
    existing_ids = {str(row["job_id"]) for row in existing_rows if row["job_id"] is not None}
    log.info("%d job_ids already in linkedin_jobs — these will be skipped", len(existing_ids))

    all_jobs = []
    for query in searches:
        try:
            found = search_linkedin_jobs(query)
            log.info("Search '%s' returned %d cards", query, len(found))
            all_jobs.extend(found)
        except Exception as exc:
            log.error("Search failed for '%s': %s", query, exc)

    all_jobs = deduplicate_jobs(all_jobs)
    new_jobs = [j for j in all_jobs if j["job_id"] not in existing_ids]
    log.info("%d unique jobs found across all searches, %d are new", len(all_jobs), len(new_jobs))

    # Cross-source dedup: skip anything that's the same role already sitting
    # in linkedin_jobs from the ATS DAG (same company + near-identical title).
    from agents.job_dedup import filter_cross_source_duplicates, RECENCY_WINDOW_DAYS
    cutoff = (datetime.utcnow() - timedelta(days=RECENCY_WINDOW_DAYS)).strftime("%Y-%m-%d")
    recent_rows = fetch_all(
        "SELECT company_name, job_title, source FROM linkedin_jobs WHERE search_date >= %s",
        (cutoff,),
    )
    new_jobs, skipped_dupes = filter_cross_source_duplicates(new_jobs, recent_rows)
    if skipped_dupes:
        log.info("Skipped %d likely cross-source duplicates (already on an ATS board)", skipped_dupes)

    context["ti"].xcom_push(key="items_attempted", value=len(searches))
    context["ti"].xcom_push(key="items_found", value=len(all_jobs))

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    for job in new_jobs:
        job["post_date"] = clean_date(job.get("post_date"))
        job["search_date"] = today_str

    for idx, job in enumerate(new_jobs):
        if idx > 0:
            time.sleep(DETAIL_FETCH_DELAY_SEC)
        description, salary = get_job_details(job["job_link"])
        job["description"] = description
        job["salary"] = salary

    context["ti"].xcom_push(key="scraped_jobs", value=new_jobs)
    log.info("Scrape complete: %d jobs ready for scoring", len(new_jobs))


def task_score_jobs(**context):
    from agents.job_agents import build_scoring_chunks, score_job_batch
    from agents.job_resume_context import RESUME_MARKDOWN, KEY_STRENGTHS_TO_WEIGHT

    jobs = context["ti"].xcom_pull(key="scraped_jobs", task_ids="search_and_scrape") or []
    if not jobs:
        log.info("No new jobs to score — skipping")
        context["ti"].xcom_push(key="enriched_jobs", value=[])
        return

    chunks = build_scoring_chunks(jobs)
    log.info("Scoring %d jobs across %d batches", len(jobs), len(chunks))

    results_by_id: dict[str, dict] = {}
    for i, chunk in enumerate(chunks):
        log.info("Scoring batch %d/%d (%d jobs)", i + 1, len(chunks), len(chunk))
        try:
            results = score_job_batch(chunk, RESUME_MARKDOWN, KEY_STRENGTHS_TO_WEIGHT)
            for r in results:
                results_by_id[r["ID"]] = r
        except Exception as exc:
            log.error("Batch %d failed to score — those %d jobs will be skipped: %s",
                      i + 1, len(chunk), exc)
        if i < len(chunks) - 1:
            time.sleep(2)  # small courtesy gap between Gemini calls

    enriched = []
    for job in jobs:
        analysis = results_by_id.get(job["job_id"])
        if not analysis:
            log.warning("No AI analysis for job_id %s (%s) — skipping insert",
                        job["job_id"], job["job_title"])
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
    log.info("Scoring complete: %d jobs enriched and ready to load", len(enriched))


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
        try:
            job_id_int = int(job["job_id"])
        except (TypeError, ValueError):
            log.warning("Skipping job with non-numeric job_id: %s", job.get("job_id"))
            continue

        row = (
            job_id_int,
            "linkedin",
            str(job_id_int),  # external_ref mirrors job_id for this source
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

    if not statements:
        log.info("Nothing valid to insert after job_id validation.")
        return

    execute_many(statements)
    log.info("Inserted %d jobs into linkedin_jobs", len(statements))


def task_log_run(**context):
    """
    Writes this run's outcome to job_scout_run_log regardless of whether
    upstream tasks succeeded (trigger_rule="all_done" below) — a scrape
    that fails outright is exactly the kind of thing the health check
    should catch, not skip over.
    """
    from agents.job_scout_health import log_run

    items_attempted = context["ti"].xcom_pull(key="items_attempted", task_ids="search_and_scrape") or 0
    items_found = context["ti"].xcom_pull(key="items_found", task_ids="search_and_scrape") or 0
    scraped = context["ti"].xcom_pull(key="scraped_jobs", task_ids="search_and_scrape") or []
    enriched = context["ti"].xcom_pull(key="enriched_jobs", task_ids="score_jobs") or []

    try:
        log_run(
            dag_id="life_os_job_scout",
            items_attempted=items_attempted,
            items_found=items_found,
            new_items=len(scraped),
            items_loaded=len(enriched),
        )
    except Exception as exc:
        log.error("Failed to write job_scout_run_log: %s", exc)


with DAG(
    dag_id="life_os_job_scout",
    default_args=default_args,
    schedule_interval="0 */6 * * *",   # every 6 hours — adjust to taste
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "jobs"],
    doc_md="""
## Job Scout DAG

Scrapes LinkedIn for the configured search titles, dedupes against jobs
already in `linkedin_jobs`, pulls full descriptions, scores fit via Gemini
2.5 Flash-Lite against the resume in `agents/job_resume_context.py`, and
inserts enriched rows into `linkedin_jobs`. The Jobs page picks these up
automatically since it reads from that same table.

**Manual trigger with a custom search list:**
```json
{"searches": ["Staff Data Engineer", "Head of Analytics"]}
```
    """,
) as dag:

    scrape = PythonOperator(task_id="search_and_scrape", python_callable=task_search_and_scrape)
    score  = PythonOperator(task_id="score_jobs",        python_callable=task_score_jobs)
    load   = PythonOperator(task_id="load_to_mysql",     python_callable=task_load_to_mysql)
    log_run_task = PythonOperator(
        task_id="log_run", python_callable=task_log_run, trigger_rule="all_done",
    )

    scrape >> score >> load >> log_run_task
