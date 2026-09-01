"""
airflow/dags/media/life_os_refresh_streaming_availability.py

Weekly refresh of MediaItem.streaming_provider_ids for TMDB-sourced items.

Background: streaming_provider_ids / streaming_fetched_at are currently
only ever set once, at add-time, in domains/media/routers/media_search.py.
If a title leaves (or joins) a streaming service afterward, the app never
finds out. This DAG closes that gap on a weekly cadence rather than
real-time, deliberately — TMDB's watch-providers endpoint has no batch
mode (confirmed: TMDB has repeatedly declined this as a feature request),
so it's one API call per title either way, and doing that on every page
load would add latency to GET /media and risk bursting past TMDB's rate
limits on a page that can render many titles at once.

Selection/priority (per project owner's own refinement of the original
proposal): don't just refresh want_to/in_progress items — spend the full
MAX_ITEMS_PER_RUN budget if there's room, filling with completed items
and then abandoned items once want_to/in_progress is exhausted. This is
implemented as a single ORDER BY + LIMIT query (priority tier, then
never-fetched-first, then oldest-fetched-first) rather than three
separate queries with manual remaining-budget bookkeeping — simpler and
can't get the budget math wrong.

*** UNVERIFIED AGAINST THE REAL airflow/dag_db.py — READ BEFORE TRUSTING ***
Per every prior DAG-touching work order in this project (see the jobs
postmortem, §8.3), airflow/dag_db.py's real interface has never been
shared with any agent working on this codebase. fetch_all()/execute() are
called here the same way the jobs DAGs' documented Phase 7 rescan-refresh
logic calls them (parameterized %s placeholders, one UPDATE per row via a
loop rather than assuming execute_many() supports differing values per
row in one batch call — that capability was never confirmed either).
Verify this against the real module before relying on it; the harness in
this same PR/commit tests against a hand-built stand-in only, per the
same methodology jobs' Phase 7 verification used.
"""

import datetime
import logging
import sys
import time

sys.path.insert(0, "/opt/airflow/project")
sys.path.insert(0, "/opt/airflow/project/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator

from dag_db import fetch_all, execute
from agents.media_agents import get_tmdb_watch_providers

log = logging.getLogger(__name__)

# ── Tunables ───────────────────────────────────────────────────────────────
REFRESH_INTERVAL_DAYS = 7    # don't re-check a title more than once a week
MAX_ITEMS_PER_RUN = 200      # hard cap on TMDB calls per DAG run
INTER_REQUEST_DELAY_SEC = 0.5  # light pacing between TMDB calls

DEFAULT_ARGS = {
    "owner": "life_os",
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}

_SELECT_CANDIDATES_SQL = """
    SELECT mi.id AS media_item_id, mi.external_id, mi.external_source,
           um.status
    FROM media_items mi
    INNER JOIN user_media um ON um.media_item_id = mi.id
    WHERE mi.external_source IN ('tmdb_movie', 'tmdb_tv')
      AND (mi.streaming_fetched_at IS NULL OR mi.streaming_fetched_at < %s)
    ORDER BY
        CASE um.status
            WHEN 'want_to'     THEN 0
            WHEN 'in_progress' THEN 0
            WHEN 'completed'   THEN 1
            WHEN 'abandoned'   THEN 2
            ELSE 3
        END,
        mi.streaming_fetched_at IS NULL DESC,
        mi.streaming_fetched_at ASC
    LIMIT %s
"""

_UPDATE_SQL = """
    UPDATE media_items
    SET streaming_provider_ids = %s,
        streaming_fetched_at = %s
    WHERE id = %s
"""


def task_select_and_refresh(**context) -> None:
    """
    Selects candidate MediaItem rows, calls TMDB for each, and pushes the
    successfully-fetched results to XCom for the next task to write.

    A per-item TMDB failure is logged and skipped, not raised — one bad
    title shouldn't fail the whole weekly run. streaming_fetched_at is
    only advanced for items that actually got a real answer back, so a
    failed item stays eligible for retry on next week's run rather than
    silently being marked "checked" without being checked.
    """
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=REFRESH_INTERVAL_DAYS)
    candidates = fetch_all(_SELECT_CANDIDATES_SQL, (cutoff, MAX_ITEMS_PER_RUN))

    log.info("Streaming refresh: %d candidate(s) selected (cap=%d)", len(candidates), MAX_ITEMS_PER_RUN)

    results = []
    errors = 0
    for i, row in enumerate(candidates):
        if i > 0:
            time.sleep(INTER_REQUEST_DELAY_SEC)

        media_item_id = row["media_item_id"]
        media_type = "movie" if row["external_source"] == "tmdb_movie" else "tv"

        try:
            provider_ids = get_tmdb_watch_providers(row["external_id"], media_type)
            results.append({"media_item_id": media_item_id, "provider_ids": provider_ids})
        except Exception as exc:
            errors += 1
            log.warning(
                "Streaming refresh: TMDB call failed for media_item_id=%s (%s/%s): %s",
                media_item_id, row["external_source"], row["external_id"], exc,
            )

    log.info("Streaming refresh: %d succeeded, %d failed", len(results), errors)
    context["ti"].xcom_push(key="refresh_results", value=results)


def task_apply_updates(**context) -> None:
    """Writes the fetched results back to media_items, one UPDATE per row."""
    results = context["ti"].xcom_pull(key="refresh_results", task_ids="select_and_refresh") or []
    now = datetime.datetime.utcnow()

    updated = 0
    for r in results:
        # provider_ids may legitimately be None (title confirmed to have
        # no US flatrate availability right now) — that's a real answer,
        # not a failure, and should overwrite stale data same as any other.
        execute(_UPDATE_SQL, (r["provider_ids"], now, r["media_item_id"]))
        updated += 1

    log.info("Streaming refresh: %d media_items updated", updated)


with DAG(
    dag_id="life_os_refresh_streaming_availability",
    description="Weekly refresh of TMDB streaming-provider availability for tracked media items",
    schedule_interval="@weekly",
    start_date=datetime.datetime(2026, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["media", "tmdb"],
) as dag:

    select_and_refresh = PythonOperator(
        task_id="select_and_refresh",
        python_callable=task_select_and_refresh,
    )

    apply_updates = PythonOperator(
        task_id="apply_updates",
        python_callable=task_apply_updates,
    )

    select_and_refresh >> apply_updates
