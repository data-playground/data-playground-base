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

DIFF STATUS (updated post-WO#13 Task 2 — this module HAS now been
diffed against the real services/tmdb_service.py; the paragraph below
described that diff as still-outstanding, which is no longer accurate):
  - US-region, flatrate-only extraction: confirmed matching.
  - 404 handling: was diverging (this module raised on 404;
    tmdb_service.py returned [] and treated it as "confirmed no
    availability") — fixed here to match.
  - media_type input contract: was diverging ("tv" only vs. "tv"/"tv_show")
    — fixed here to accept "tv_show" too, matching tmdb_service.py's
    public contract.
  - provider_id extraction: this module defensively filters entries
    missing "provider_id"; tmdb_service.py does not. Left as-is — this
    module being more defensive isn't a bug worth "fixing away."
  If either implementation changes again, re-diff the other — this module
  still intentionally duplicates rather than imports tmdb_service.py's
  logic (see the GOVERNANCE §2.2 note above), so drift can silently
  reappear.

ORIGINAL CAVEAT (kept for history — see DIFF STATUS above for current
state): this module's request/response handling was written from general
knowledge of TMDB's public v3 API shape, not by reading
services/tmdb_service.py's actual implementation — that file was never
shared with any agent in this engagement. The same caution job_agents.py's
get_full_job_posting() flagged for its unverified LinkedIn selectors (see
migration_docs/Work Orders/work_order_03_jobs_domain.md's postmortem,
§Phase 3) applied here until the diff above was actually done.
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
        media_type: "movie", "tv", or "tv_show". "tv_show" is accepted and
            mapped to the same "tv" URL segment as "tv" — this matches
            MediaItem.media_type's actual stored value ("tv_show", per
            media_recommend.py's RecommendationMediaType) and
            services/tmdb_service.py's own public contract (which accepts
            "movie"/"tv_show" and does the same internal mapping). Fixed
            post-WO#13 Task 2: this module previously required the
            literal "tv" and raised ValueError on "tv_show", which never
            broke anything only because this function's one current
            caller (life_os_refresh_streaming_availability.py) happens to
            always convert to "movie"/"tv" first — but was a footgun for
            any future caller passing MediaItem.media_type directly.

    Returns:
        A list of TMDB provider IDs (ints), or None if the title has no
        US flatrate availability recorded by TMDB right now (a title that
        used to be on a service and no longer is will correctly come back
        as None/empty here — that's the whole point of this DAG existing).

    Raises:
        ValueError: if media_type is not "movie", "tv", or "tv_show".
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
    if media_type not in ("movie", "tv", "tv_show"):
        raise ValueError(f"media_type must be 'movie', 'tv', or 'tv_show', got {media_type!r}")

    url_type = "tv" if media_type in ("tv", "tv_show") else "movie"
    url = f"{TMDB_API_BASE}/{url_type}/{tmdb_id}/watch/providers"
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
