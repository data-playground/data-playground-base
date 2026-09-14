# airflow/dags/medium/life_os_medium_detect_source.py
"""
On-demand DAG for classifying and adding a single tracked source
(WO#35 follow-up). Triggered from the settings page's "Add a source"
form — not scheduled (schedule=None) — with the pasted URL passed
through the DAG run's `conf`.

Deliberately does NOT use a separate "pending request" table. Airflow's
own run history already gives exactly what was wanted here — every
attempt, its conf, its success/failure, and a retry button — without
duplicating that state in the app's database. The only durable output
of a successful run is the new row this task inserts directly into
medium_feed_sources; a failed run leaves no row at all, which is itself
the "did this work" signal. There's no in-between "pending" state to
track because nothing needs to poll for one — you'd check Airflow, not
the settings page, to see how it went.

Not retried automatically (retries=0, unlike life_os_medium_ingest.py's
transient-503-style retry) — a failed detection is almost always either
a genuinely wrong URL or a feed that doesn't exist, not a transient
blip. Re-run manually (clear the task instance, or just resubmit the
form) once you've looked at why it failed.

Same GOVERNANCE.md §2.2 boundary as every other DAG here: never imports
models.py, database.py, or any router/service. dag_db.execute() is
confirmed against the real dag_db.py (see life_os_medium_ingest.py's
docstring for that confirmation).
"""
from __future__ import annotations

import datetime
import logging
import sys

sys.path.insert(0, "/opt/airflow/project")
sys.path.insert(0, "/opt/airflow/project/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator

from dag_db import execute

from domains.medium.rss_ingest import identify_source

log = logging.getLogger(__name__)

_INSERT_SQL = (
    "INSERT INTO medium_feed_sources (source_type, identifier, label, is_active, created_at) "
    "VALUES (%s, %s, %s, 1, %s)"
)


def task_detect_source(**context) -> None:
    dag_run = context.get("dag_run")
    conf = (dag_run.conf if dag_run is not None else None) or {}
    raw_url = conf.get("raw_url")
    label = conf.get("label")

    if not raw_url:
        raise ValueError("No raw_url in the triggering conf — nothing to detect.")

    result = identify_source(raw_url)

    if not result.matched:
        attempted_summary = "; ".join(f"{url} -> {outcome}" for url, outcome in result.attempted)
        raise RuntimeError(
            f"Could not identify a working feed for {raw_url!r}. "
            f"Tried: {attempted_summary or 'nothing (input shape matched no candidate type)'}"
        )

    now = datetime.datetime.now(datetime.timezone.utc)
    # A duplicate (source_type, identifier) fails here on the table's own
    # unique constraint — deliberately not pre-checked; the DB's own
    # "Duplicate entry" error in the task log already says exactly that.
    execute(_INSERT_SQL, (result.source_type.value, result.identifier, label, now))

    log.info(
        "Added medium_feed_sources row: %s / %s (%d sample article(s): %s)",
        result.source_type.value, result.identifier, len(result.sample_titles), result.sample_titles,
    )


default_args = {
    "owner": "life-os",
    "retries": 0,  # see module docstring — not a transient-failure kind of task
}


def _register_dag() -> DAG:
    with DAG(
        dag_id="life_os_medium_detect_source",
        default_args=default_args,
        schedule=None,  # trigger-only, fired from the settings page — never runs on its own
        start_date=datetime.datetime(2026, 9, 13),
        catchup=False,
        tags=["medium", "detection"],
    ) as dag:
        PythonOperator(
            task_id="detect_source",
            python_callable=task_detect_source,
        )
    return dag


if __name__ != "__main__":
    dag = _register_dag()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python life_os_medium_detect_source.py <medium_url> [label]")
        raise SystemExit(1)

    class _FakeDagRun:
        conf = {"raw_url": sys.argv[1], "label": sys.argv[2] if len(sys.argv) > 2 else None}

    print(f"Running detect_source manually for {sys.argv[1]!r}...")
    task_detect_source(dag_run=_FakeDagRun())
    print("Done.")
