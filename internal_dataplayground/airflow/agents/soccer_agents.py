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

FIELD NAMES. The calendar and /live endpoints name things differently:
the calendar uses Home/Away (team name under a TeamName locale list),
/live uses HomeTeam/AwayTeam. Field names here were verified against
captured real payloads (a Qatar-vs-Ecuador World Cup match for the
calendar shape; finished league and cup matches for /live). MatchStatus
meanings come from the _MATCH_STATUS_MAP table below. Everything is
still read defensively (dict.get(), never direct indexing) — that
guards against missing/malformed data but does NOT validate field
names, so check any new field against a real response first. A missing
field or unrecognized status code never blocks ingestion: the row gets
NULLs / status_label="unknown", and the full raw JSON stays in
soccer_raw_payloads for reprocessing.
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
    ingest DAG — the watch list lives in soccer_competitions (seeded by
    the s0cc3r_d0ma1n001 migration, edited on /soccer/settings). This is
    the "fetch competitions from FIFA" admin flow — the same role ats_slug_service.py plays for the jobs
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

# FIFA MatchStatus -> label mapping.
#
# Authoritative mapping (confirmed 2026-09-10 against a known reference):
#   0 -> Completed
#   1 -> Scheduled
#   3 -> Decided on Penalties  (a completed match — bucketed as "finished",
#        not "live")
#   7 -> Postponed             (distinct from "scheduled" — no longer
#        expected at its listed kickoff time, but not resolved either)
#   9 -> Forfeited/Suspended   (bucketed as "finished" so the details/
#        events backfill task treats it as terminal and stops re-fetching
#        it forever. Caveat: "Suspended" specifically could mean a match
#        was stopped mid-play and may resume later, which "finished"
#        would get wrong — no separate code was given to distinguish that
#        case from a genuine forfeit, so this is a known, accepted
#        imprecision rather than an oversight.)
#
# No code for "currently in progress" has been confirmed yet. Every code
# not listed here — including any of 2/4/5/6/8, and any future code FIFA
# might use — falls through to "unknown" rather than a guess. A previous
# version of this map confidently guessed labels for untested codes; that
# was wrong and is exactly the failure mode this map is now written to
# avoid.
_MATCH_STATUS_MAP = {
    0: "finished",
    1: "scheduled",
    3: "finished",
    7: "postponed",
    9: "finished",
}


def _extract_localized_text(obj, key: str = "Name") -> Optional[str]:
    """
    FIFA embeds lots of things this way — team names, stadium names, and
    apparently others — as a localized list under `key`, shaped like
    {"Name": [{"Locale": "en-GB", "Description": "Maracanã"}, ...]}.
    This pulls the English entry out defensively.

    CONFIRMED BUG #1 (production, 2026-09-10): the original version of
    this function only existed as `_extract_team_name()` and was applied
    to HomeTeam/AwayTeam only. `Stadium.Name` turned out to use the exact
    same localized-list shape, but venue_name was reading `stadium.get("Name")`
    directly — so it stored the raw list-of-dicts instead of a string,
    and pymysql crashed trying to escape a dict as a SQL parameter. Fixed
    by sharing this one function across every caller.

    CONFIRMED BUG #2 (production, 2026-09-10, same day): every
    HomeTeam/AwayTeam on the calendar/matches endpoint came back as
    "TBD" in the UI — i.e. this function returned None for all of them —
    while HomeTeamScore/AwayTeamScore populated correctly as plain
    integers. That strongly suggests FIFA's calendar-list endpoint gives
    team (and likely stadium) names as PLAIN STRINGS, not the nested
    localized-list object this function originally assumed everywhere
    (that nested shape may be real, but only on the richer /live detail
    endpoint, not this summary one). The `isinstance(obj, str)` branch
    below handles that directly. This is inferred from behavior, not
    from a captured sample — if it's still wrong, pull one raw match
    object from soccer_raw_payloads and we'll fix it against real data
    instead of a third guess.

    Never returns anything but a str or None — never the raw list/dict —
    as a defense-in-depth measure; see parse_match_summary()'s
    _scalar_or_none() guard for the same protection applied generically.
    """
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return None
    values = obj.get(key)
    if isinstance(values, list):
        for entry in values:
            if isinstance(entry, dict) and str(entry.get("Locale", "")).startswith("en"):
                desc = entry.get("Description")
                return desc if isinstance(desc, str) else None
        if values and isinstance(values[0], dict):
            desc = values[0].get("Description")
            return desc if isinstance(desc, str) else None
        return None
    if isinstance(values, str):
        return values
    # Team objects sometimes carry a flatter TeamName/Description field
    # instead of the localized Name list.
    fallback = obj.get("TeamName") or obj.get("Description")
    return fallback if isinstance(fallback, str) else None


def _scalar_or_none(value):
    """
    Last line of defense before a value goes into parse_match_summary()'s
    return dict, which the DAG binds directly as SQL parameters (see
    dag_db.py::execute()). If FIFA hands back a nested dict/list where a
    plain string or number was expected — as venue_name did in
    production before this fix — a MySQL driver can't escape that as a
    parameter and the whole ingest task dies. Coercing to None instead
    means a field-shape surprise degrades to a NULL column, not a failed
    task. See _extract_localized_text()'s docstring for the incident this
    is responding to.
    """
    if isinstance(value, (dict, list)):
        return None
    return value


def parse_match_summary(raw_match: dict) -> dict:
    """
    Extracts the normalized fields SoccerMatch needs from one raw FIFA
    calendar-endpoint match dict. Every field is read defensively — see
    the module docstring's FIELD-NAME CAVEAT. Missing fields resolve to
    None rather than raising, so a shape surprise degrades to a thinner
    row instead of failing the whole ingest run. Every field that should
    be a plain scalar is also passed through _scalar_or_none() as a
    second line of defense — see that function's docstring.
    """
    status_code = raw_match.get("MatchStatus")
    try:
        status_code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status_code = None

    home_obj = raw_match.get("Home") or {}
    away_obj = raw_match.get("Away") or {}

    return {
        "fifa_competition_id": str(raw_match.get("IdCompetition", "")),
        "fifa_season_id":      str(raw_match.get("IdSeason", "")),
        "fifa_stage_id":       str(raw_match.get("IdStage", "")),
        "fifa_match_id":       str(raw_match.get("IdMatch", "")),
        "home_team_name":      _scalar_or_none(_extract_localized_text(home_obj, key="TeamName")),
        "away_team_name":      _scalar_or_none(_extract_localized_text(away_obj, key="TeamName")),
        # Confirmed present on both Home/Away (calendar endpoint) and
        # HomeTeam/AwayTeam (/live endpoint) — used to build real crest
        # image URLs (https://api.fifa.com/api/v3/picture/teams-{format}-
        # {size}/{IdTeam}). Populated here rather than waiting for the
        # /live fetch, so a crest can render even for a scheduled,
        # not-yet-finished match.
        "fifa_home_team_id":   _scalar_or_none(home_obj.get("IdTeam")),
        "fifa_away_team_id":   _scalar_or_none(away_obj.get("IdTeam")),
        "home_team_score":     _scalar_or_none(raw_match.get("HomeTeamScore")),
        "away_team_score":     _scalar_or_none(raw_match.get("AwayTeamScore")),
        "kickoff_at":          raw_match.get("Date"),  # ISO string — DAG parses to datetime
        "fifa_match_status_code": status_code,
        "status_label":        _MATCH_STATUS_MAP.get(status_code, "unknown"),
        "venue_name":          _scalar_or_none(_extract_localized_text(raw_match.get("Stadium"))),
    }


# ── ENDPOINT 3 PARSER — LINEUPS, GOALS, BOOKINGS, SUBS, COACHES ──────────────
#
# Everything below is parsed from the /live detail endpoint (NOT the
# calendar endpoint above — note the different top-level key,
# HomeTeam/AwayTeam here vs Home/Away on the calendar endpoint; FIFA is
# genuinely inconsistent between its own endpoints, confirmed by direct
# comparison of real payloads from both).
#
# Unlike parse_match_summary() above, this parser is built against a
# real, fully-populated, finished match (Bournemouth 2-2 Brentford,
# 2026-09-12) rather than guessed field shapes — every field name below
# is confirmed, not inferred by analogy. Two things are still genuinely
# unconfirmed and flagged where relevant: the red-card Bookings.Card
# code (only yellow, code 1, was observed), and Coaches.Role's exact
# meaning (0/1 pattern held for both teams independently, which is
# decent evidence, but it's two data points, not a spec).

def _first_locale_text(locale_list) -> Optional[str]:
    """
    Same idea as _extract_localized_text(), but for fields that are
    directly a locale list themselves (e.g. PlayerName), not a dict
    wrapping one under a named key. FIFA mixes "en-GB" and "en-gb"
    casing across different parts of the same payload — confirmed in
    this exact match's data — so the locale match is case-insensitive.
    """
    if not isinstance(locale_list, list):
        return None
    for entry in locale_list:
        if isinstance(entry, dict) and str(entry.get("Locale", "")).lower().startswith("en"):
            desc = entry.get("Description")
            if isinstance(desc, str):
                return desc
    if locale_list and isinstance(locale_list[0], dict):
        desc = locale_list[0].get("Description")
        return desc if isinstance(desc, str) else None
    return None


def _parse_minute(minute_str: Optional[str]) -> Optional[int]:
    """
    "45'+4'" -> 49, "38'" -> 38. Stoppage time is added to the base
    minute for a single sortable/placeable number — the verbatim string
    is kept separately (minute_display) for on-screen display, since
    "49'" would misrepresent a stoppage-time goal as a 49th-minute one.
    """
    if not minute_str:
        return None
    cleaned = minute_str.replace("'", "")
    parts = [p for p in cleaned.split("+") if p.strip()]
    try:
        return sum(int(p) for p in parts)
    except ValueError:
        return None


# FIFA Period code for a penalty shootout (confirmed: a real shootout
# match's Goals array carried period-11 rows). Shootout kicks are excluded from the goals list in
# parse_match_lineup_data() so scorer lines, pitch badges and assist
# counts only reflect goals scored in play; the shootout result is carried
# by home_penalty_score / away_penalty_score instead. The raw payload in
# soccer_raw_payloads still holds the individual kicks.
_SHOOTOUT_PERIOD_CODE = 11


def _is_shootout_period(period) -> bool:
    """True if a Goals[].Period value is the shootout code (int or numeric str)."""
    try:
        return int(period) == _SHOOTOUT_PERIOD_CODE
    except (TypeError, ValueError):
        return False


def parse_match_lineup_data(raw_details: dict) -> dict:
    """
    Parses a /live match-detail payload into normalized lineup/goal/
    booking/substitution/coach rows, plus a handful of scalar match
    fields (formation, possession, attendance). Intended to be called
    only once a match is confirmed 'finished' — see
    life_os_soccer_ingest.py::parse_finished_match_details().

    Goals with Period == _SHOOTOUT_PERIOD_CODE (penalty-shootout kicks)
    are omitted from "goals"; the shootout result is in
    home_penalty_score / away_penalty_score.

    Returns a dict with keys: home_formation, away_formation,
    possession_home, possession_away, attendance, lineups, goals,
    bookings, substitutions, coaches — the last five are lists of dicts
    ready to bind as SQL parameters.
    """
    home = raw_details.get("HomeTeam") or {}
    away = raw_details.get("AwayTeam") or {}
    possession = raw_details.get("BallPossession") or {}

    def _players(team: dict, side: str) -> list[dict]:
        rows = []
        for p in (team.get("Players") or []):
            rows.append({
                "team_side": side,
                "fifa_player_id": str(p.get("IdPlayer", "")),
                "shirt_number": _scalar_or_none(p.get("ShirtNumber")),
                "player_name": _first_locale_text(p.get("PlayerName")),
                # Confirmed: 0=GK, 1=DEF, 2=AM/wide, 3=FWD, 6=DM.
                "position_code": _scalar_or_none(p.get("Position")),
                "is_starter": p.get("Status") == 1,
                "is_captain": bool(p.get("Captain")),
                "on_field_at_finish": p.get("FieldStatus") == 1,
            })
        return rows

    def _goals(team: dict, side: str) -> list[dict]:
        rows = []
        for g in (team.get("Goals") or []):
            # Penalty-shootout kicks are not goals for scorer/badge
            # purposes — see _SHOOTOUT_PERIOD_CODE.
            if _is_shootout_period(g.get("Period")):
                continue
            rows.append({
                "team_side": side,
                "fifa_player_id": str(g.get("IdPlayer", "")) if g.get("IdPlayer") else None,
                "fifa_assist_player_id": str(g["IdAssistPlayer"]) if g.get("IdAssistPlayer") else None,
                "minute_display": g.get("Minute"),
                "minute_numeric": _parse_minute(g.get("Minute")),
                "period": _scalar_or_none(g.get("Period")),
            })
        return rows

    def _bookings(team: dict, side: str) -> list[dict]:
        rows = []
        for b in (team.get("Bookings") or []):
            rows.append({
                "team_side": side,
                "fifa_player_id": str(b.get("IdPlayer", "")) if b.get("IdPlayer") else None,
                "card_type_code": _scalar_or_none(b.get("Card")),
                "minute_display": b.get("Minute"),
                "minute_numeric": _parse_minute(b.get("Minute")),
                "period": _scalar_or_none(b.get("Period")),
                "reason": b.get("Reason") if isinstance(b.get("Reason"), str) else None,
            })
        return rows

    def _subs(team: dict, side: str) -> list[dict]:
        rows = []
        for s in (team.get("Substitutions") or []):
            rows.append({
                "team_side": side,
                "fifa_player_off_id": str(s.get("IdPlayerOff", "")) if s.get("IdPlayerOff") else None,
                "fifa_player_on_id": str(s.get("IdPlayerOn", "")) if s.get("IdPlayerOn") else None,
                "player_off_name": _first_locale_text(s.get("PlayerOffName")),
                "player_on_name": _first_locale_text(s.get("PlayerOnName")),
                "minute_display": s.get("Minute"),
                "minute_numeric": _parse_minute(s.get("Minute")),
                "period": _scalar_or_none(s.get("Period")),
            })
        return rows

    def _coaches(team: dict, side: str) -> list[dict]:
        rows = []
        for c in (team.get("Coaches") or []):
            rows.append({
                "team_side": side,
                "fifa_coach_id": str(c.get("IdCoach", "")) if c.get("IdCoach") else None,
                "name": _first_locale_text(c.get("Name")),
                "role_code": _scalar_or_none(c.get("Role")),
            })
        return rows

    return {
        "home_formation": _scalar_or_none(home.get("Tactics")),
        "away_formation": _scalar_or_none(away.get("Tactics")),
        # FIFA's own image-URL TEMPLATE for the team, e.g.
        # ".../picture/flags-{format}-{size}/KSA" — the {format}/{size}
        # placeholders are filled in at render time (lineup_helpers.py).
        "home_team_picture_url": _scalar_or_none(home.get("PictureUrl")),
        "away_team_picture_url": _scalar_or_none(away.get("PictureUrl")),
        "possession_home": _scalar_or_none(possession.get("OverallHome")),
        "possession_away": _scalar_or_none(possession.get("OverallAway")),
        "attendance": _scalar_or_none(raw_details.get("Attendance")),
        # Confirmed real top-level fields — see the module's 2026-09-18
        # note above on why these were added (a real shootout match).
        "home_penalty_score": _scalar_or_none(raw_details.get("HomeTeamPenaltyScore")),
        "away_penalty_score": _scalar_or_none(raw_details.get("AwayTeamPenaltyScore")),
        "lineups": _players(home, "home") + _players(away, "away"),
        "goals": _goals(home, "home") + _goals(away, "away"),
        "bookings": _bookings(home, "home") + _bookings(away, "away"),
        "substitutions": _subs(home, "home") + _subs(away, "away"),
        "coaches": _coaches(home, "home") + _coaches(away, "away"),
    }


# ── EVENT-STREAM STATS (from /timelines) ──────────────────────────────────
#
# NOTE: shootout outcomes live in home_penalty_score / away_penalty_score
# (see parse_match_lineup_data()); individual shootout kicks are excluded
# from the goals list (_SHOOTOUT_PERIOD_CODE).
#
# This module also derives a handful of match stats from
# the /timelines endpoint (a different raw payload than match_details —
# see fetch_match_events() / the "match_events" endpoint in
# soccer_raw_payloads). Confirmed, structured Type codes only — no
# English-text matching of EventDescription, which would be a
# meaningfully weaker foundation:
#   12 = "Attempt at Goal" (shot)
#   15 = "Offside"
#   16 = "Corner"
#   18 = "Foul"
# Deliberately NOT counted here: shots on target, saves and blocks.
# Observed in a real payload: keeper saves are Type 57 (attributed to the
# DEFENDING team) and outfield blocks are Type 17, i.e. distinct codes —
# no text matching needed. Goals are Type 0. How these relate to Type 12
# (whether a saved shot also appears as a Type 12 event) is unverified;
# see postmortem item I2 before deriving "shots on target".

_EVENT_TYPE_SHOT = 12
_EVENT_TYPE_OFFSIDE = 15
_EVENT_TYPE_CORNER = 16
_EVENT_TYPE_FOUL = 18


def parse_match_event_stats(raw_events: list, fifa_home_team_id: str, fifa_away_team_id: str) -> dict:
    """
    Tallies shots/offsides/corners/fouls per side from a /timelines
    payload (a flat list of event dicts — see fetch_match_events()).
    Events with no IdTeam (kickoff/half-time/full-time markers) or an
    IdTeam matching neither side are silently skipped rather than
    raising — defensive, same posture as the rest of this module.

    Returns a dict with keys shots_home/shots_away/offsides_home/
    offsides_away/corners_home/corners_away/fouls_home/fouls_away, all
    ints (0 if the event type never occurred, not None — an empty
    /timelines payload, e.g. for a match not yet started, returns all
    zeros rather than missing keys).
    """
    counts = {
        "shots_home": 0, "shots_away": 0,
        "offsides_home": 0, "offsides_away": 0,
        "corners_home": 0, "corners_away": 0,
        "fouls_home": 0, "fouls_away": 0,
    }
    for event in (raw_events or []):
        if not isinstance(event, dict):
            continue
        team_id = str(event.get("IdTeam", ""))
        if team_id == fifa_home_team_id:
            side = "home"
        elif team_id == fifa_away_team_id:
            side = "away"
        else:
            continue

        event_type = event.get("Type")
        if event_type == _EVENT_TYPE_SHOT:
            counts[f"shots_{side}"] += 1
        elif event_type == _EVENT_TYPE_OFFSIDE:
            counts[f"offsides_{side}"] += 1
        elif event_type == _EVENT_TYPE_CORNER:
            counts[f"corners_{side}"] += 1
        elif event_type == _EVENT_TYPE_FOUL:
            counts[f"fouls_{side}"] += 1

    return counts
