# airflow/dags/medium/life_os_medium_ingest.py
"""
Daily ingestion DAG for the medium domain (WO#35).

Reads the tracked-sources list from medium_feed_sources (managed
entirely via /medium/settings — see
domains/medium/routers/medium_settings.py), fetches each active
source's current RSS feed through
domains.medium.rss_ingest.ingest_sources(), and upserts the results
into medium_articles.

Per CONTRIBUTING.md / GOVERNANCE.md §2.2 — same boundary
life_os_nba_ingest.py follows — this file never imports models.py,
database.py, or any router/service. All database access goes through
airflow/dag_db.py.

Confirmed against the real dag_db.py (shared after this file was first
written): fetch_all(sql: str, params: tuple = ()) -> list[dict] and
execute_many(statements) both match exactly what's used below — no
changes needed. dag_db.py itself contains what looks like two
implementations back to back (an apparently-unused SQLAlchemy
get_sync_engine()/get_sync_session() pair, then the raw-pymysql
get_connection()/fetch_all()/execute_many() actually used here and by
life_os_nba_ingest.py) — not this file's concern, just noted in case
it's worth cleaning up separately.

No date-range/backfill mode here, unlike life_os_nba_ingest.py —
Medium's RSS feeds only ever expose current state (the most recent
~10-25 items per source), never a historical date. There's no such
thing as backfilling old articles through RSS at all, so this DAG has
exactly one behavior: pull whatever's in each active source's feed
right now, upsert it. Re-running same-day is always safe — every
write is an idempotent upsert on `guid`.
"""
from __future__ import annotations

import datetime
import json
import logging
import sys

sys.path.insert(0, "/opt/airflow/project")
sys.path.insert(0, "/opt/airflow/project/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator

from dag_db import execute_many, fetch_all  # ← both confirmed against the real dag_db.py

from domains.medium.rss_ingest import FeedSource, FeedSourceType, ingest_sources

log = logging.getLogger(__name__)

_ARTICLE_COLUMNS = [
    "guid", "source_type", "source_identifier", "source_label",
    "title", "url", "author", "published_at", "tags",
    "summary", "content_html", "raw_item", "fetched_at",
]


def _rows_to_sources_and_labels(
    rows: list[dict],
) -> tuple[list[FeedSource], dict[tuple[str, str], str | None]]:
    """Turns medium_feed_sources rows into what ingest_sources() expects
    (plain FeedSource values, no label) plus a side lookup so the label
    can be re-attached to each article at upsert time. Split out from
    _load_active_sources() specifically so this mapping logic — unlike
    the DB read around it — is testable without a live database."""
    sources: list[FeedSource] = []
    labels: dict[tuple[str, str], str | None] = {}
    for row in rows:
        try:
            source_type = FeedSourceType(row["source_type"])
        except ValueError:
            log.warning("Skipping medium_feed_sources row with unknown source_type=%r", row["source_type"])
            continue
        sources.append(FeedSource(type=source_type, identifier=row["identifier"]))
        labels[(row["source_type"], row["identifier"])] = row.get("label")
    return sources, labels


def _load_active_sources() -> tuple[list[FeedSource], dict[tuple[str, str], str | None]]:
    rows = fetch_all("SELECT source_type, identifier, label FROM medium_feed_sources WHERE is_active = 1")
    return _rows_to_sources_and_labels(rows)


def _build_upsert_statements(articles, labels: dict[tuple[str, str], str | None]) -> list[tuple[str, tuple]]:
    """Builds the (sql, params) pairs execute_many() takes — same
    INSERT ... ON DUPLICATE KEY UPDATE shape as life_os_nba_ingest.py's
    _upsert(), adapted to medium_articles' single natural key (`guid`)
    instead of a composite one. Pure — no dag_db call in here — purely
    so this part is testable in isolation from the DB.

    `summary` is always written empty: rss_ingest.ParsedArticle doesn't
    populate a distinct summary today (content:encoded or <description>
    both land in content_html) — the column exists on medium_articles
    for future use, not silently faked here.
    """
    if not articles:
        return []
    placeholders = ", ".join(["%s"] * len(_ARTICLE_COLUMNS))
    update_clause = ", ".join(f"{c}=VALUES({c})" for c in _ARTICLE_COLUMNS if c != "guid")
    sql = (
        f"INSERT INTO medium_articles ({', '.join(_ARTICLE_COLUMNS)}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_clause}"
    )

    statements = []
    for a in articles:
        label = labels.get((a.source_type.value, a.source_identifier))
        statements.append((sql, (
            a.guid,
            a.source_type.value,
            a.source_identifier,
            label,
            a.title,
            a.url,
            a.author,
            a.published_at,
            json.dumps(a.tags),
            "",  # summary — see docstring above
            a.content_html,
            a.raw_item,
            a.fetched_at,
        )))
    return statements


def _upsert_articles(articles, labels: dict[tuple[str, str], str | None]) -> int:
    statements = _build_upsert_statements(articles, labels)
    if not statements:
        return 0
    execute_many(statements)
    log.info("Upserted %d article(s) into medium_articles", len(statements))
    return len(statements)


def task_ingest_articles(**context) -> None:
    """The only function both the scheduled DAG task and the CLI entry
    point below call — no second, parallel implementation to drift out
    of sync, same principle life_os_nba_ingest.py follows."""
    sources, labels = _load_active_sources()
    if not sources:
        log.info("No active medium_feed_sources rows — nothing to ingest.")
        return

    articles = ingest_sources(sources)
    _upsert_articles(articles, labels)
    log.info(
        "Medium ingest complete: %d active source(s), %d article(s) upserted.",
        len(sources), len(articles),
    )


default_args = {
    "owner": "life-os",
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


def _register_dag() -> DAG:
    """Builds the actual Airflow DAG object — same reasoning as
    life_os_nba_ingest.py's _register_dag() for why this only happens
    on Airflow's own import, never on a direct CLI run."""
    with DAG(
        dag_id="life_os_medium_ingest",
        default_args=default_args,
        schedule="0 6 * * *",  # 6am UTC — arbitrary, no dependency on any other job; change freely
        start_date=datetime.datetime(2026, 9, 13),
        catchup=False,
        tags=["medium", "ingestion"],
    ) as dag:
        PythonOperator(
            task_id="ingest_articles",
            python_callable=task_ingest_articles,
        )
    return dag


if __name__ != "__main__":
    dag = _register_dag()

if __name__ == "__main__":
    print("Running medium ingest manually (same task the scheduled DAG runs)...")
    task_ingest_articles()
    print("Done.")
