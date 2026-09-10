# airflow/agents/soccer_agents.py
"""
airflow/agents/soccer_agents.py

DAG-facing FIFA API client for the soccer domain (WO#34).

Adapted from a source script (fifa_data_pull.py) originally written
against FIFA's public api.fifa.com / cxm-api.fifa.com endpoints and
loading results into BigQuery. This module keeps the request logic
(URLs, headers, pagination via continuation token/hash, the details/
playbyplay endpoint shapes) unchanged from that script — only the
BigQuery loading code was dropped, since Life OS's ingest DAG writes to
MariaDB via dag_db.py instead (see
airflow/dags/soccer/life_os_soccer_ingest.py).

No API key or auth token is used — FIFA's public site calls these
endpoints directly from the browser. The headers below (User-Agent,
sec-ch-ua, Origin) mimic a browser request; this is the same "public
JSON API, no login" situation as job_ats_agents.py's Greenhouse/Lever
fetchers, not a credentialed integration, so there is nothing to put in
GCP Secret Manager here.

ARCHITECTURAL NOTE: like job_agents.py, job_ats_agents.py, and
media_agents.py, this file is imported by DAG tasks only. It has no
dependency on models.py, database.py, or any FastAPI router/service, in
line with the DAG/FastAPI boundary rule in CONTRIBUTING.md /
GOVERNANCE.md §2.2.

FIELD-NAME CAVEAT (same spirit as media_agents.py's original caveat
before its WO#13 diff-and-fix pass): the source script never captured a
sample match/detail/event JSON body, so the field names this module
reads off calendar-endpoint results (Date, HomeTeam, AwayTeam,
MatchStatus, etc. — see parse_match_summary()) are inferred from general
knowledge of FIFA's public API shape, not verified against a captured
response. Every read is defensive (dict.get(), never direct indexing)
specifically because of this. VERIFY AGAINST A LIVE RESPONSE before
trusting status_label/scores in production — see _MATCH_STATUS_MAP's own
docstring. Nothing here blocks ingestion if a field is missing or a
status code is unrecognized: worst case, a match row gets NULL scores or
status_label="unknown", and the full raw JSON is still preserved in
soccer_raw_payloads for reprocessing later once the real field names/
codes are confirmed.
"""
import logging
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

_TIMEOUT = 20
_RETRY_BACKOFF = [2, 5, 10]  # seconds, applied between attempts

_CALENDAR_URL = "https://api.fifa.com/api/v3/calendar/matches"
_COMPETITIONS_URL = (
    "https://cxm-api.fifa.com/fifaplusweb/api/sections/matches/"
    "competitionslist/3jlHZVPUI0eeeBFTX9qSZ1?locale=en"
)
_LIVE_URL_TMPL = "https://api.fifa.com/api/v3/live/football/{competition}/{season}/{stage}/{match}?language=en"
_TIMELINE_URL_TMPL = "https://api.fifa.com/api/v3/timelines/{competition}/{season}/{stage}/{match}?language=en"

# Base headers copied verbatim from the source script — FIFA's API
# appears to expect a browser-like User-Agent/sec-ch-ua set. `Host` is
# overridden per-call since api.fifa.com and cxm-api.fifa.com are
# different hosts.
_BASE_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9,pt;q=0.8",
    "Connection": "keep-alive",
    "Host": "api.fifa.com",
    "Origin": "https://www.fifa.com",
    "sec-ch-ua": '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "Windows",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
    ),
}


def _get_with_retry(url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> dict:
    """
    Shared GET-with-retry for all FIFA calls. FIFA's public endpoints have
    no documented rate limit (there's no key to attach a quota to), but
    they're a public site's backing API — retry on transient failures
    rather than hammering immediately, the same conservative posture as
    ats_slug_service.py's probes.
    """
    last_exc = None
    for attempt, wait in enumerate([0] + _RETRY_BACKOFF):
        if wait:
            time.sleep(wait)
        try:
            resp = requests.get(url, params=params, headers=headers or _BASE_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            log.warning(
                "FIFA GET failed (attempt %d/%d) for %s: %s",
                attempt + 1, len(_RETRY_BACKOFF) + 1, url, exc,
            )
    raise RuntimeError(f"FIFA API unavailable after retries: {url}. Last error: {last_exc}")


# ── ENDPOINT 1 — COMPETITIONS LIST ───────────────────────────────────────────

# In-process cache — the /soccer/settings page's live search box calls
# fetch_competitions() on every keystroke (see
# domains/soccer/routers/soccer_settings.py::search_fifa_competitions),
# and this list runs 200+ entries but changes rarely. Same pattern as
# tmdb_service.py's _genre_cache: a plain module-level dict, no external
# invalidation beyond "restart clears it" or the TTL below expiring.
_COMPETITIONS_CACHE_TTL_SEC = 6 * 60 * 60  # 6 hours
_competitions_cache: dict = {"data": None, "fetched_at": 0.0}


def fetch_competitions() -> list[dict]:
    """
    Fetches FIFA's full competitions list (200+ entries), cached
    in-process for _COMPETITIONS_CACHE_TTL_SEC. Not called by the daily
    ingest DAG — the watch list lives in soccer_competitions, seeded by
    life_os_soccer_ingest.py's _SEED_COMPETITIONS. This is the "quick
    button to fetch the competition list from FIFA" admin flow (WO#34
    fast-follow) — the same role ats_slug_service.py plays for the jobs
    domain, just backed by a real FIFA endpoint instead of a guess-probe.

    competitionId is cast to str — see the module docstring's rationale
    on why FIFA competition IDs are never treated as integers.
    """
    now = time.time()
    cached = _competitions_cache["data"]
    if cached is not None and (now - _competitions_cache["fetched_at"]) < _COMPETITIONS_CACHE_TTL_SEC:
        return cached

    headers = _BASE_HEADERS.copy()
    headers["Host"] = "cxm-api.fifa.com"
    data = _get_with_retry(_COMPETITIONS_URL, headers=headers)
    competitions = data.get("competitions", [])
    result = [
        {"competitionId": str(c.get("competitionId")), "name": c.get("name")}
        for c in competitions
        if c.get("competitionId") is not None
    ]
    _competitions_cache["data"] = result
    _competitions_cache["fetched_at"] = now
    return result


# ── ENDPOINT 2 — MATCH CALENDAR (fixtures + results) ─────────────────────────

def fetch_matches(competition_id: str, from_date: str, to_date: str) -> list[dict]:
    """
    Fetches every match for one competition between two dates, following
    FIFA's continuation-token pagination (500 rows/page max) exactly as
    the source script did.

    Args:
        competition_id: FIFA competition ID, as a string (see module
            docstring on why this is never cast to int).
        from_date, to_date: "YYYY-MM-DD" — converted to FIFA's expected
            "...T00:00:00Z" / "...T23:59:59Z" format internally.

    Returns:
        List of raw match dicts, verbatim from FIFA (each still has
        IdCompetition/IdSeason/IdStage/IdMatch plus whatever score/team/
        status fields FIFA includes — see parse_match_summary() for how
        those get read defensively).
    """
    from_ts = f"{from_date}T00:00:00Z"
    to_ts = f"{to_date}T23:59:59Z"

    all_matches: list[dict] = []
    continuation_token = None
    continuation_hash = None

    while True:
        headers = _BASE_HEADERS.copy()
        headers["X-Mdp-Continuation-Token"] = continuation_token
        params = {
            "from": from_ts,
            "to": to_ts,
            "language": "en",
            "count": 500,
            "idCompetition": competition_id,
            "continuationhash": continuation_hash,
        }
        data = _get_with_retry(_CALENDAR_URL, params=params, headers=headers)
        page = data.get("Results", [])
        all_matches.extend(page)

        continuation_token = data.get("ContinuationToken")
        continuation_hash = data.get("ContinuationHash")
        if not continuation_hash:
            break

    log.info(
        "Fetched %d matches for competition %s (%s to %s)",
        len(all_matches), competition_id, from_date, to_date,
    )
    return all_matches


# ── ENDPOINT 3 & 4 — MATCH DETAILS + PLAY-BY-PLAY ────────────────────────────

def fetch_match_details(competition_id: str, season_id: str, stage_id: str, match_id: str) -> dict:
    """Fetches the /live detail payload for one match (players, officials, stadium, score)."""
    url = _LIVE_URL_TMPL.format(competition=competition_id, season=season_id, stage=stage_id, match=match_id)
    return _get_with_retry(url)


def fetch_match_events(competition_id: str, season_id: str, stage_id: str, match_id: str) -> list[dict]:
    """
    Fetches the /timelines play-by-play payload for one match. Older
    matches have sparse events (just goals) — that's a FIFA data
    limitation carried over from the source script's own comment, not a
    bug in this function.
    """
    url = _TIMELINE_URL_TMPL.format(competition=competition_id, season=season_id, stage=stage_id, match=match_id)
    data = _get_with_retry(url)
    return data.get("Event", [])


# ── PARSING HELPERS (defensive — see module docstring's FIELD-NAME CAVEAT) ──

# Best-effort FIFA MatchStatus -> label mapping. UNVERIFIED against a
# live response (see module docstring). Codes not listed here fall
# through to "unknown" rather than raising, so an unrecognized code never
# blocks ingestion — worst case is a match stuck at
# status_label="unknown" until this map is corrected against real data.
_MATCH_STATUS_MAP = {
    0: "scheduled",
    1: "live",      # first half
    2: "live",      # half-time
    3: "live",      # second half
    5: "finished",
    6: "finished",  # abandoned — treated as finished so it stops re-fetching
    7: "scheduled", # postponed
    8: "finished",  # cancelled — treated as finished so it stops re-fetching
}


def _extract_team_name(team_obj) -> Optional[str]:
    """
    FIFA team objects are typically shaped like
    {"Name": [{"Locale": "en-GB", "Description": "Brazil"}], ...} — pull
    the English description defensively. Falls back to None rather than
    raising if the shape doesn't match (see FIELD-NAME CAVEAT).
    """
    if not isinstance(team_obj, dict):
        return None
    names = team_obj.get("Name")
    if isinstance(names, list):
        for entry in names:
            if isinstance(entry, dict) and str(entry.get("Locale", "")).startswith("en"):
                return entry.get("Description")
        if names and isinstance(names[0], dict):
            return names[0].get("Description")
    return team_obj.get("TeamName") or team_obj.get("Description")


def parse_match_summary(raw_match: dict) -> dict:
    """
    Extracts the normalized fields SoccerMatch needs from one raw FIFA
    calendar-endpoint match dict. Every field is read defensively — see
    the module docstring's FIELD-NAME CAVEAT. Missing fields resolve to
    None rather than raising, so a shape surprise degrades to a thinner
    row instead of failing the whole ingest run.
    """
    status_code = raw_match.get("MatchStatus")
    try:
        status_code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status_code = None

    stadium = raw_match.get("Stadium")
    venue_name = stadium.get("Name") if isinstance(stadium, dict) else None

    return {
        "fifa_competition_id": str(raw_match.get("IdCompetition", "")),
        "fifa_season_id":      str(raw_match.get("IdSeason", "")),
        "fifa_stage_id":       str(raw_match.get("IdStage", "")),
        "fifa_match_id":       str(raw_match.get("IdMatch", "")),
        "home_team_name":      _extract_team_name(raw_match.get("HomeTeam")),
        "away_team_name":      _extract_team_name(raw_match.get("AwayTeam")),
        "home_team_score":     raw_match.get("HomeTeamScore"),
        "away_team_score":     raw_match.get("AwayTeamScore"),
        "kickoff_at":          raw_match.get("Date"),  # ISO string — DAG parses to datetime
        "fifa_match_status_code": status_code,
        "status_label":        _MATCH_STATUS_MAP.get(status_code, "unknown"),
        "venue_name":          venue_name,
    }
