"""
airflow/agents/media_agents.py

DAG-facing TMDB helper for the media domain. Exists because DAGs never
import from services/ (GOVERNANCE.md §2.2 — this rule predates the
domain-folder governance pass and is treated as absolute). services/
tmdb_service.py already has a get_streaming_providers()-shaped function
used by domains/media/routers/media_search.py, but this module does NOT
import or wrap it — it makes its own independent call, the same way
job_agents.py calls LinkedIn/Greenhouse directly rather than importing a
router-side service.

IMPORTANT CAVEAT, stated plainly rather than glossed over: this module's
request/response handling was written from general knowledge of TMDB's
public v3 API shape, not by reading services/tmdb_service.py's actual
implementation — that file was never shared with any agent in this
engagement. It is plausible (arguably likely) that the two
implementations diverge in some detail: which regions/streaming types
count (this one only extracts "flatrate" — subscription-included
availability — for the US region, matching MediaItem's own docstring
("available for streaming in the US"), not one-off rent/buy offers),
error handling, or retry behavior. Diff this against the real
services/tmdb_service.py before trusting it in production, the same way
job_agents.py's get_full_job_posting() had its LinkedIn selectors flagged
as unverified pending a live check (see migration_docs/Work Orders/
work_order_03_jobs_domain.md's postmortem, §Phase 3).
"""

import logging
import os

import requests

log = logging.getLogger(__name__)

TMDB_API_BASE = "https://api.themoviedb.org/3"
REQUEST_TIMEOUT_SEC = 10


def _tmdb_key() -> str:
    """
    Matches the env-var-only pattern used throughout this codebase's
    agent modules (e.g. os.environ.get("GEMINI_API") in blog_agents.py) —
    docker-compose.yml already passes TMDB_API_KEY to both the
    airflow-webserver and airflow-scheduler containers, so no compose
    change is needed for this DAG to read it.
    """
    key = os.environ.get("TMDB_API_KEY")
    if not key:
        raise RuntimeError("TMDB_API_KEY is not set in the Airflow container's environment")
    return key


def get_tmdb_watch_providers(tmdb_id: str, media_type: str) -> list[int] | None:
    """
    Fetches current US "flatrate" (subscription) streaming providers for
    one TMDB title.

    Args:
        tmdb_id: The TMDB numeric ID (MediaItem.external_id for
            tmdb_movie/tmdb_tv sourced items).
        media_type: "movie" or "tv".

    Returns:
        A list of TMDB provider IDs (ints), or None if the title has no
        US flatrate availability recorded by TMDB right now (a title that
        used to be on a service and no longer is will correctly come back
        as None/empty here — that's the whole point of this DAG existing).

    Raises:
        requests.HTTPError: on a non-2xx, non-404 TMDB response, so callers
            can distinguish "confirmed no availability" (empty list or None)
            from "the request itself failed" (exception) and handle each
            differently — the DAG task below only advances
            streaming_fetched_at on a successful call, so a transient
            TMDB outage doesn't get misread as "nothing streams this
            anywhere" and silently overwrite good prior data.

    NOTE (fixed post-WO#13 Task 2 finding): TMDB returns 404 on this
    endpoint for titles it has no watch-provider data for at all — a real,
    confirmed answer, not a request failure. This is special-cased below
    to return None rather than raise, matching
    services/tmdb_service.py's get_streaming_providers() behavior on the
    same status code (that function does `if resp.status_code == 404:
    return []` before raise_for_status()). Before this fix, a 404 here
    raised requests.HTTPError, which the refresh DAG's except-block
    treated as a transient failure and silently skipped — meaning
    streaming_fetched_at was never advanced for titles TMDB legitimately
    404s on, and they were retried forever instead of being recorded as
    "confirmed: no availability" the way the add-time path already does.
    """
    if media_type not in ("movie", "tv"):
        raise ValueError(f"media_type must be 'movie' or 'tv', got {media_type!r}")

    url = f"{TMDB_API_BASE}/{media_type}/{tmdb_id}/watch/providers"
    resp = requests.get(
        url,
        params={"api_key": _tmdb_key()},
        timeout=REQUEST_TIMEOUT_SEC,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()

    us_results = (data.get("results") or {}).get("US") or {}
    flatrate = us_results.get("flatrate") or []
    provider_ids = [p["provider_id"] for p in flatrate if "provider_id" in p]

    return provider_ids or None
