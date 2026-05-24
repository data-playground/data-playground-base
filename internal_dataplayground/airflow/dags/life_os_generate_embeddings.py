# airflow/dags/life_os_generate_embeddings.py
"""
Embedding Generation DAG
─────────────────────────────────────────────────────────────────────────────
Runs nightly at 2:00 AM. Generates sentence-transformer embeddings for any
media_item where:
  - embedding IS NULL (never embedded), OR
  - embedding_generated_at is older than 7 days (stale — metadata may have changed)

The model runs in the dedicated ml-service container.
This DAG calls ml-service via HTTP — it does NOT import sentence-transformers.

Pipeline:
  1. Pre-flight: check ml-service is healthy.
  2. Fetch all media_items needing embeddings.
  3. Process in batches of 50 — build embedding text, call /embed, store results.
  4. Log summary.

Can also be triggered manually from Airflow UI with optional conf:
  {"force_all": true}  — re-embeds ALL items regardless of age
  {"media_item_ids": [1, 2, 3]}  — embeds specific items only

Model: all-MiniLM-L6-v2 (384-dim), running in ml-service container.
"""

import sys
import logging
from datetime import datetime, timedelta

sys.path.insert(0, '/opt/airflow/project')
sys.path.insert(0, '/opt/airflow/project/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

BATCH_SIZE = 50
STALE_DAYS = 7

default_args = {
    "owner": "life_os",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


def task_preflight(**context):
    """
    Checks that the ml-service container is healthy before starting.
    Raises if unavailable — prevents the main task from running blind.
    """
    from services.ml_service_client import health_check_sync

    if not health_check_sync():
        raise RuntimeError(
            "ML service is not healthy. "
            "Check that the ml-service container is running: "
            "docker compose ps ml-service"
        )
    log.info("ML service health check passed.")


def task_generate_embeddings(**context):
    """
    Fetches media_items needing embeddings and generates them in batches.

    For each item:
      1. Build embedding text from title + genres + description + author.
      2. Call ml-service /embed endpoint.
      3. Store the 384-dim vector in media_items.embedding.
      4. Set embedding_generated_at = NOW().
    """
    from dag_db import fetch_all, execute_many
    from services.ml_service_client import embed_texts_sync, build_embedding_text

    conf = context["dag_run"].conf or {}
    force_all = conf.get("force_all", False)
    specific_ids = conf.get("media_item_ids")

    import json
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    stale_cutoff = (now - timedelta(days=STALE_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # ── Fetch items needing embeddings ─────────────────────────────────────────
    if specific_ids:
        ids_placeholder = ','.join(['%s'] * len(specific_ids))
        items = fetch_all(
            f"SELECT id, title, media_type, genres, description, author "
            f"FROM media_items WHERE id IN ({ids_placeholder})",
            tuple(specific_ids),
        )
        log.info("Forced embedding for %d specific items.", len(items))

    elif force_all:
        items = fetch_all(
            "SELECT id, title, media_type, genres, description, author FROM media_items"
        )
        log.info("Force-all mode: embedding %d items.", len(items))

    else:
        items = fetch_all(
            "SELECT id, title, media_type, genres, description, author "
            "FROM media_items "
            "WHERE embedding IS NULL "
            "   OR embedding_generated_at < %s",
            (stale_cutoff,),
        )
        log.info("Found %d items needing embeddings (NULL or stale > %d days).",
                 len(items), STALE_DAYS)

    if not items:
        log.info("Nothing to embed — all items are up to date.")
        return

    # ── Process in batches ─────────────────────────────────────────────────────
    total_success = 0
    total_failed = 0

    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start: batch_start + BATCH_SIZE]
        log.info(
            "Processing batch %d-%d of %d...",
            batch_start + 1,
            min(batch_start + BATCH_SIZE, len(items)),
            len(items),
        )

        # Build embedding texts
        texts = []
        for item in batch:
            genres = []
            if item.get("genres"):
                try:
                    g = item["genres"]
                    if isinstance(g, str):
                        g = json.loads(g)
                    genres = g if isinstance(g, list) else []
                except Exception:
                    genres = []

            texts.append(build_embedding_text({
                "title":       item["title"] or "",
                "genres":      genres,
                "description": item.get("description") or "",
                "author":      item.get("author") or "",
                "media_type":  item.get("media_type") or "",
            }))

        # Call ML service
        try:
            embeddings = embed_texts_sync(texts)
        except Exception as exc:
            log.error("ML service call failed for batch at %d: %s", batch_start, exc)
            total_failed += len(batch)
            continue

        if len(embeddings) != len(batch):
            log.error(
                "Embedding count mismatch: expected %d, got %d. Skipping batch.",
                len(batch), len(embeddings),
            )
            total_failed += len(batch)
            continue

        # Store results
        statements = []
        for item, embedding in zip(batch, embeddings):
            embedding_json = json.dumps(embedding)
            statements.append((
                "UPDATE media_items SET embedding = %s, embedding_generated_at = %s WHERE id = %s",
                (embedding_json, now_str, item["id"]),
            ))

        try:
            from dag_db import execute_many as _execute_many
            _execute_many(statements)
            total_success += len(batch)
            log.info("Batch stored: %d embeddings.", len(batch))
        except Exception as exc:
            log.error("Failed to store batch at %d: %s", batch_start, exc)
            total_failed += len(batch)

    log.info(
        "Embedding generation complete. Success: %d | Failed: %d | Total: %d",
        total_success, total_failed, len(items),
    )

    if total_failed > 0:
        raise RuntimeError(
            f"{total_failed} items failed embedding. Check logs above."
        )


def task_refresh_streaming_providers(**context):
    """
    Refreshes streaming provider data for items where streaming_fetched_at
    is older than 7 days or NULL. TMDB data changes as licensing deals shift.

    Runs after embeddings so providers are always fresh alongside embeddings.
    Only processes TMDB items (movies and TV shows) — books have no providers.
    """
    import os
    import json
    import asyncio
    from dag_db import fetch_all, execute

    # We need async for the TMDB service — run in a sync event loop
    async def _refresh():
        import sys
        sys.path.insert(0, '/opt/airflow/project')
        from services import tmdb_service

        stale_cutoff = (datetime.utcnow() - timedelta(days=STALE_DAYS)).strftime("%Y-%m-%d %H:%M:%S")

        items = fetch_all(
            "SELECT id, external_id, media_type FROM media_items "
            "WHERE external_source IN ('tmdb_movie', 'tmdb_tv') "
            "AND external_id IS NOT NULL "
            "AND (streaming_fetched_at IS NULL OR streaming_fetched_at < %s)",
            (stale_cutoff,),
        )

        if not items:
            log.info("No streaming provider data to refresh.")
            return

        log.info("Refreshing streaming providers for %d items.", len(items))
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        success = 0

        for item in items:
            try:
                provider_ids = await tmdb_service.get_streaming_providers(
                    tmdb_id=item["external_id"],
                    media_type=item["media_type"],
                )
                providers_json = json.dumps(provider_ids) if provider_ids else None
                execute(
                    "UPDATE media_items SET streaming_provider_ids = %s, streaming_fetched_at = %s WHERE id = %s",
                    (providers_json, now_str, item["id"]),
                )
                success += 1
            except Exception as exc:
                log.warning("Could not refresh providers for item %d: %s", item["id"], exc)

        log.info("Streaming provider refresh complete: %d/%d updated.", success, len(items))

    asyncio.run(_refresh())


with DAG(
    dag_id="life_os_generate_embeddings",
    default_args=default_args,
    schedule_interval="0 2 * * *",   # 2:00 AM daily
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "media", "ml"],
    doc_md="""
## Embedding Generation DAG

Runs nightly at 2:00 AM. Generates 384-dim sentence-transformer embeddings
for new or stale media_items using the ml-service container.

Also refreshes TMDB streaming provider data (which services have each item).

**Manual trigger options:**
```json
{"force_all": true}
{"media_item_ids": [1, 2, 3]}
```

**Pre-requisite:** ml-service container must be running.
`docker compose ps ml-service`
    """,
) as dag:

    preflight = PythonOperator(
        task_id="preflight_ml_service",
        python_callable=task_preflight,
    )

    generate = PythonOperator(
        task_id="generate_embeddings",
        python_callable=task_generate_embeddings,
    )

    refresh_streaming = PythonOperator(
        task_id="refresh_streaming_providers",
        python_callable=task_refresh_streaming_providers,
    )

    preflight >> generate >> refresh_streaming
