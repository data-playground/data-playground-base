# airflow/agents/nba_agents.py
"""
NBA data extraction (WO#33) — ported from the standalone nba_game_summary.py
notebook script the project owner supplied.

Deliberately pure: HTTP + response-shape parsing only, no database code of
any kind. Per CONTRIBUTING.md / GOVERNANCE.md §2.2, DAG files never import
models.py/database.py/routers/services, and all DAG database access goes
through airflow/dag_db.py's raw-SQL helpers — so this module hands the DAG
plain dicts/lists, and airflow/dags/nba/*.py is the only place that turns
those into SQL. Nothing here imports dag_db either, which keeps this module
independently testable and, in principle, reusable from a router someday
(it isn't today — v1 has no live-fetch page, everything reads pre-ingested
rows from MariaDB) without the DAG/FastAPI boundary ever being in question.

What changed vs. the original script, and why:
  - The BigQuery-schema-generation function (func_create_schema_from_skeleton)
    is gone entirely — MariaDB doesn't use it, and models.py's flat tables
    replace what it was for. The recursive skeleton *filter* function
    (func_filter_data_to_skeleton) is kept, because the "fields" skeletons
    it filters against are still the shortest way to describe each
    endpoint's shape and drop everything we don't store.
  - The class-with-mutable-run-state (`NBA`) pattern is gone. Every fetcher
    here is a plain function; nothing carries state across calls except the
    module-level request-throttle timestamp below.
  - A minimum delay between requests was added (see _throttle()). The
    original script had none — looping ~10 box-score endpoints x N games
    with zero pause is exactly the pattern that gets rate-limited or
    blocked, which is part of what happened to the enrichment step this
    build deliberately does not port forward (see module-end note).
  - The Gemini-based `enrich_results()` narrative-recap step is NOT ported.
    It was never turned on in the original and, per the project owner, did
    not reliably work. If it's picked back up later, it must go through
    services/ai/call_gemini_json() (GOVERNANCE.md §2.3), not a raw
    genai.Client() call — see models.py for the intentionally-unused hook
    this would attach to.
"""
from __future__ import annotations

import logging
import random
import time
from datetime import date, datetime

import requests

log = logging.getLogger(__name__)


# ── HTTP ──────────────────────────────────────────────────────────────────────
# Headers ported verbatim from nba_game_summary.py. Both hosts reject
# requests without a browser-like User-Agent; core-api.nba.com additionally
# requires the subscription key below, which is NBA's own public web-client
# key (embedded in nba.com's front-end JS, not a credential of ours) — it
# stays a plain constant here rather than moving to GCP Secret Manager /
# services.ai.keys, since it isn't actually a secret we control.
STATS_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
    "Connection": "keep-alive",
    "Referer": "https://stats.nba.com/",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}

CORE_API_HEADERS = {
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "ocp-apim-subscription-key": "747fa6900c6c4e89a58b81b72f36eb96",
    "Origin": "https://www.nba.com",
    "priority": "u=1, i",
    "referer": "https://www.nba.com/",
    "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
}

# Minimum delay between outbound requests (plus jitter). New in this port —
# see module docstring. Tuned conservatively for a personal-scale nightly
# job (a handful of games), not for throughput.
_MIN_REQUEST_INTERVAL_SECONDS = 0.6
_last_request_at = 0.0


def _throttle() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < _MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(_MIN_REQUEST_INTERVAL_SECONDS - elapsed + random.uniform(0, 0.25))
    _last_request_at = time.monotonic()


def _fetch_json(url: str, headers: dict, retries: int = 3, timeout: float = 20.0) -> dict:
    """GET with the throttle above, plus basic retry/backoff on 429/503/timeouts."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        _throttle()
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code in (429, 503):
                wait = (attempt + 1) * 5
                log.warning("NBA API HTTP %s on attempt %d/%d — waiting %ds",
                            resp.status_code, attempt + 1, retries, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            wait = 10 * (attempt + 1)
            log.warning("NBA API request failed (attempt %d/%d): %s — waiting %ds",
                        attempt + 1, retries, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"NBA API unavailable after {retries} retries: {last_exc}")


# ── Parsing helpers (ported from the NBA class's func_* methods) ───────────────

def _filter_data_to_skeleton(raw_data, skeleton):
    """Recursively filters raw JSON to match a skeleton's shape. v3 endpoints."""
    if isinstance(skeleton, dict):
        source = raw_data or {}
        return {k: _filter_data_to_skeleton(source.get(k), v) for k, v in skeleton.items()}
    if isinstance(skeleton, list):
        if skeleton and isinstance(raw_data, list):
            if raw_data:
                return [_filter_data_to_skeleton(item, skeleton[0]) for item in raw_data]
            return [{k: None for k in skeleton[0]}]
        return []
    return raw_data


def _filter_results_by_list(results: list[dict], headers: list[str], name: str) -> list[dict]:
    """Reads the resultSets[{name,headers,rowSet}] shape used by v2 endpoints."""
    idx = [r["name"] for r in results].index(name)
    headers_loc = [results[idx]["headers"].index(h) for h in headers]
    return [
        {h: row[pos] for h, pos in zip(headers, headers_loc)}
        for row in results[idx]["rowSet"]
    ]


def _get_items_in_list_of_dicts(results: list[dict], headers: list[str], new_headers: dict) -> list[dict]:
    """
    Reads the modules[0].cards[].cardData shape used by GAME_DATA (v4).
    Uses .get() rather than the original's operator.itemgetter() so a card
    missing a field (seen occasionally on preseason/all-star cards) yields
    None for that field instead of crashing the whole discovery call.
    """
    out = []
    for r in results:
        card = r.get("cardData", {})
        out.append({new_headers[h]: card.get(h) for h in headers})
    return out


# ── Endpoint catalog ─────────────────────────────────────────────────────────

def _player_box_score_skeleton(stat_keys: list[str]) -> dict:
    """
    Builds the standard "one row per player per game" v3 skeleton shared by
    nine of the ten box-score endpoints, parameterized only by which
    statistics block that endpoint returns. See BOX_SCORE_STAT_MAPS below —
    each category's source->column mapping doubles as this skeleton's key
    list, so the two never drift out of sync.
    """
    inner = {"personId": None, "position": None, "statistics": {k: None for k in stat_keys}}
    return {"gameId": None, "homeTeam": {"teamId": None, "players": [inner]},
            "awayTeam": {"teamId": None, "players": [inner]}}


# Source-JSON-key -> target-DB-column, one dict per box-score variant.
# These are also used to build each endpoint's parsing skeleton (see
# _player_box_score_skeleton) and by flatten_player_box_score() below —
# defining a stat category's shape once, in one place.
BOX_SCORE_STAT_MAPS: dict[str, dict[str, str]] = {
    "BS_TRAD": {
        "minutes": "minutes", "fieldGoalsMade": "field_goals_made",
        "fieldGoalsAttempted": "field_goals_attempted", "fieldGoalsPercentage": "field_goal_percentage",
        "threePointersMade": "three_pointers_made", "threePointersAttempted": "three_pointers_attempted",
        "threePointersPercentage": "three_pointer_percentage", "freeThrowsMade": "free_throws_made",
        "freeThrowsAttempted": "free_throws_attempted", "freeThrowsPercentage": "free_throw_percentage",
        "reboundsOffensive": "rebounds_offensive", "reboundsDefensive": "rebounds_defensive",
        "reboundsTotal": "rebounds_total", "assists": "assists", "steals": "steals", "blocks": "blocks",
        "turnovers": "turnovers", "foulsPersonal": "fouls_personal", "points": "points",
        "plusMinusPoints": "plus_minus",
    },
    "BS_ADV": {
        "minutes": "minutes", "estimatedOffensiveRating": "estimated_offensive_rating",
        "offensiveRating": "offensive_rating", "estimatedDefensiveRating": "estimated_defensive_rating",
        "defensiveRating": "defensive_rating", "estimatedNetRating": "estimated_net_rating",
        "netRating": "net_rating", "assistPercentage": "assist_percentage",
        "assistToTurnover": "assist_to_turnover", "assistRatio": "assist_ratio",
        "offensiveReboundPercentage": "offensive_rebound_percentage",
        "defensiveReboundPercentage": "defensive_rebound_percentage",
        "reboundPercentage": "rebound_percentage", "turnoverRatio": "turnover_ratio",
        "effectiveFieldGoalPercentage": "effective_field_goal_percentage",
        "trueShootingPercentage": "true_shooting_percentage", "usagePercentage": "usage_percentage",
        "estimatedUsagePercentage": "estimated_usage_percentage", "estimatedPace": "estimated_pace",
        "pace": "pace", "pacePer40": "pace_per_40", "possessions": "possessions", "PIE": "pie",
    },
    "BS_MISC": {
        "minutes": "minutes", "pointsOffTurnovers": "points_off_turnovers",
        "pointsSecondChance": "points_second_chance", "pointsFastBreak": "points_fast_break",
        "pointsPaint": "points_paint", "oppPointsOffTurnovers": "opp_points_off_turnovers",
        "oppPointsSecondChance": "opp_points_second_chance", "oppPointsFastBreak": "opp_points_fast_break",
        "oppPointsPaint": "opp_points_paint", "blocks": "blocks", "blocksAgainst": "blocks_against",
        "foulsPersonal": "fouls_personal", "foulsDrawn": "fouls_drawn",
    },
    "BS_SCORE": {
        "minutes": "minutes",
        "percentageFieldGoalsAttempted2pt": "pct_field_goals_attempted_2pt",
        "percentageFieldGoalsAttempted3pt": "pct_field_goals_attempted_3pt",
        "percentagePoints2pt": "pct_points_2pt", "percentagePointsMidrange2pt": "pct_points_midrange_2pt",
        "percentagePoints3pt": "pct_points_3pt", "percentagePointsFastBreak": "pct_points_fast_break",
        "percentagePointsFreeThrow": "pct_points_free_throw",
        "percentagePointsOffTurnovers": "pct_points_off_turnovers",
        "percentagePointsPaint": "pct_points_paint", "percentageAssisted2pt": "pct_assisted_2pt",
        "percentageUnassisted2pt": "pct_unassisted_2pt", "percentageAssisted3pt": "pct_assisted_3pt",
        "percentageUnassisted3pt": "pct_unassisted_3pt", "percentageAssistedFGM": "pct_assisted_fgm",
        "percentageUnassistedFGM": "pct_unassisted_fgm",
    },
    "BS_USAGE": {
        "minutes": "minutes", "usagePercentage": "usage_percentage",
        "percentageFieldGoalsMade": "pct_field_goals_made",
        "percentageFieldGoalsAttempted": "pct_field_goals_attempted",
        "percentageThreePointersMade": "pct_three_pointers_made",
        "percentageThreePointersAttempted": "pct_three_pointers_attempted",
        "percentageFreeThrowsMade": "pct_free_throws_made",
        "percentageFreeThrowsAttempted": "pct_free_throws_attempted",
        "percentageReboundsOffensive": "pct_rebounds_offensive",
        "percentageReboundsDefensive": "pct_rebounds_defensive",
        "percentageReboundsTotal": "pct_rebounds_total", "percentageAssists": "pct_assists",
        "percentageTurnovers": "pct_turnovers", "percentageSteals": "pct_steals",
        "percentageBlocks": "pct_blocks", "percentageBlocksAllowed": "pct_blocks_allowed",
        "percentagePersonalFouls": "pct_personal_fouls",
        "percentagePersonalFoulsDrawn": "pct_personal_fouls_drawn", "percentagePoints": "pct_points",
    },
    "BS_FOUR": {
        "minutes": "minutes", "effectiveFieldGoalPercentage": "effective_field_goal_percentage",
        "freeThrowAttemptRate": "free_throw_attempt_rate",
        "teamTurnoverPercentage": "team_turnover_percentage",
        "offensiveReboundPercentage": "offensive_rebound_percentage",
        "oppEffectiveFieldGoalPercentage": "opp_effective_field_goal_percentage",
        "oppFreeThrowAttemptRate": "opp_free_throw_attempt_rate",
        "oppTeamTurnoverPercentage": "opp_team_turnover_percentage",
        "oppOffensiveReboundPercentage": "opp_offensive_rebound_percentage",
    },
    "BS_TRACK": {
        "minutes": "minutes", "distance": "distance",
        "reboundChancesOffensive": "rebound_chances_offensive",
        "reboundChancesDefensive": "rebound_chances_defensive",
        "reboundChancesTotal": "rebound_chances_total", "touches": "touches",
        "secondaryAssists": "secondary_assists", "freeThrowAssists": "free_throw_assists",
        "passes": "passes", "assists": "assists",
        "contestedFieldGoalsMade": "contested_field_goals_made",
        "contestedFieldGoalsAttempted": "contested_field_goals_attempted",
        "contestedFieldGoalPercentage": "contested_field_goal_percentage",
        "uncontestedFieldGoalsMade": "uncontested_field_goals_made",
        "uncontestedFieldGoalsAttempted": "uncontested_field_goals_attempted",
        "uncontestedFieldGoalsPercentage": "uncontested_field_goal_percentage",
        "fieldGoalPercentage": "field_goal_percentage",
        "defendedAtRimFieldGoalsMade": "defended_at_rim_field_goals_made",
        "defendedAtRimFieldGoalsAttempted": "defended_at_rim_field_goals_attempted",
        "defendedAtRimFieldGoalPercentage": "defended_at_rim_field_goal_percentage",
    },
    "BS_HUSTLE": {
        "minutes": "minutes", "points": "points", "contestedShots": "contested_shots",
        "contestedShots2pt": "contested_shots_2pt", "contestedShots3pt": "contested_shots_3pt",
        "deflections": "deflections", "chargesDrawn": "charges_drawn",
        "screenAssists": "screen_assists", "screenAssistPoints": "screen_assist_points",
        "looseBallsRecoveredOffensive": "loose_balls_recovered_offensive",
        "looseBallsRecoveredDefensive": "loose_balls_recovered_defensive",
        "looseBallsRecoveredTotal": "loose_balls_recovered_total",
        "offensiveBoxOuts": "offensive_box_outs", "defensiveBoxOuts": "defensive_box_outs",
        "boxOutPlayerTeamRebounds": "box_out_player_team_rebounds",
        "boxOutPlayerRebounds": "box_out_player_rebounds", "boxOuts": "box_outs",
    },
    "BS_DEF": {
        "matchupMinutes": "matchup_minutes", "partialPossessions": "partial_possessions",
        "switchesOn": "switches_on", "playerPoints": "player_points",
        "defensiveRebounds": "defensive_rebounds", "matchupAssists": "matchup_assists",
        "matchupTurnovers": "matchup_turnovers", "steals": "steals", "blocks": "blocks",
        "matchupFieldGoalsMade": "matchup_field_goals_made",
        "matchupFieldGoalsAttempted": "matchup_field_goals_attempted",
        "matchupFieldGoalPercentage": "matchup_field_goal_percentage",
        "matchupThreePointersMade": "matchup_three_pointers_made",
        "matchupThreePointersAttempted": "matchup_three_pointers_attempted",
        "matchupThreePointerPercentage": "matchup_three_pointer_percentage",
    },
}

_MATCHUP_STAT_KEYS = [
    "matchupMinutes", "matchupMinutesSort", "PartialPossessions", "percentageDefenderTotalTime",
    "percentageOffensiveTotalTime", "percentageTotalTimeBothOn", "switchesOn", "playerPoints",
    "teamPoints", "matchupAssists", "matchupPotentialAssists", "matchupTurnovers", "matchupBlocks",
    "matchupFieldGoalsMade", "matchupFieldGoalsAttempted", "matchupFieldGoalsPercentage",
    "matchupThreePointersMade", "matchupThreePointersAttempted", "matchupThreePointersPercentage",
    "helpBlocks", "helpFieldGoalsMade", "helpFieldGoalsAttempted", "helpFieldGoalsPercentage",
    "matchupFreeThrowsMade", "matchupFreeThrowsAttempted", "shootingFouls",
]
_BS_MATCH_SKELETON = {
    "gameId": None,
    "homeTeam": {"teamId": None, "players": [{
        "personId": None, "position": None,
        "matchups": [{"personId": None, "statistics": {k: None for k in _MATCHUP_STAT_KEYS}}],
    }]},
    "awayTeam": {"teamId": None, "players": [{
        "personId": None, "position": None,
        "matchups": [{"personId": None, "statistics": {k: None for k in _MATCHUP_STAT_KEYS}}],
    }]},
}

ENDPOINTS: dict[str, dict] = {
    "PLAYERS": {
        "endpoint": "https://stats.nba.com/stats/commonallplayers?",
        "version": 2,
        "table_name": "CommonAllPlayers",
        "fields": ["PERSON_ID", "DISPLAY_FIRST_LAST", "ROSTERSTATUS", "FROM_YEAR", "TO_YEAR", "TEAM_ID"],
        # commonallplayers with IsOnlyCurrentSeason=0 (every player since
        # 1946, several thousand rows) reliably read-timed-out at 20s from
        # our Airflow host in production — bumped well past what the
        # current-season-only call (fetch_all_players()'s default) needs,
        # as headroom for the rare full-history sync too.
        "timeout": 45.0,
    },
    "GAMES": {
        "endpoint": "https://stats.nba.com/stats/leaguegamefinder?",
        "version": 2,
        "table_name": "LeagueGameFinderResults",
        "fields": ["SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE", "WL"],
    },
    "GAME_DATA": {
        "endpoint": "https://core-api.nba.com/cp/api/v1.9/feeds/gamecardfeed?",
        "version": 4,
        "fields": ["gameId", "gameStatus", "gameStatusText"],
        "rename_fields": {"gameId": "GAME_ID", "gameStatus": "GAME_STATUS_ID", "gameStatusText": "GAME_STATUS_TEXT"},
    },
    "BS_SUMMARY": {
        "endpoint": "https://stats.nba.com/stats/boxscoresummaryv3?",
        "version": 3,
        "table_name": "boxScoreSummary",
        "fields": {
            "gameId": None, "gameStatus": None, "gameStatusText": None, "period": None,
            "gameEt": None, "duration": None, "attendance": None, "sellout": None,
            "gameLabel": None, "gameSubLabel": None, "seriesText": None, "isNeutral": None,
            "arena": {"arenaName": None, "arenaCity": None, "arenaState": None},
            "homeTeam": {"teamId": None, "teamWins": None, "teamLosses": None, "score": None},
            "awayTeam": {"teamId": None, "teamWins": None, "teamLosses": None, "score": None},
        },
    },
    "PBP": {
        "endpoint": "https://stats.nba.com/stats/playbyplayv3?",
        "version": 3,
        "table_name": "game",
        "fields": {"gameId": None, "actions": [{
            "actionNumber": None, "actionId": None, "clock": None, "period": None, "teamId": None,
            "teamTricode": None, "personId": None, "playerName": None, "xLegacy": None, "yLegacy": None,
            "shotDistance": None, "shotResult": None, "isFieldGoal": None, "scoreHome": None,
            "scoreAway": None, "pointsTotal": None, "location": None, "description": None,
            "actionType": None, "subType": None, "shotValue": None,
        }]},
    },
    "BS_MATCH": {
        "endpoint": "https://stats.nba.com/stats/boxscorematchupsv3?",
        "version": 3,
        "table_name": "boxScoreMatchups",
        "fields": _BS_MATCH_SKELETON,
    },
}

# The other nine box-score variants share one shape (see
# _player_box_score_skeleton) — built here from BOX_SCORE_STAT_MAPS so the
# endpoint URL is the only thing that varies per entry.
# NOTE: BS_HUSTLE and BS_DEF's URLs say "v2" while every sibling says "v3" —
# that's NBA's own endpoint naming, carried over unchanged from
# nba_game_summary.py, not a typo introduced here. Both still return the
# v3-style nested shape, hence `"version": 3` below for both.
_BOX_SCORE_URLS = {
    "BS_TRAD":   ("https://stats.nba.com/stats/boxscoretraditionalv3?", "boxScoreTraditional"),
    "BS_ADV":    ("https://stats.nba.com/stats/boxscoreadvancedv3?", "boxScoreAdvanced"),
    "BS_MISC":   ("https://stats.nba.com/stats/boxScoreMiscv3?", "boxScoreMisc"),
    "BS_SCORE":  ("https://stats.nba.com/stats/boxscorescoringv3?", "boxScoreScoring"),
    "BS_USAGE":  ("https://stats.nba.com/stats/boxscoreusagev3?", "boxScoreUsage"),
    "BS_FOUR":   ("https://stats.nba.com/stats/boxscorefourfactorsv3?", "boxScoreFourFactors"),
    "BS_TRACK":  ("https://stats.nba.com/stats/boxscoreplayertrackv3?", "boxScorePlayerTrack"),
    "BS_HUSTLE": ("https://stats.nba.com/stats/boxscorehustlev2?", "boxScoreHustle"),
    "BS_DEF":    ("https://stats.nba.com/stats/boxscoredefensivev2?", "boxScoreDefensive"),
}
for _key, (_url, _table_name) in _BOX_SCORE_URLS.items():
    ENDPOINTS[_key] = {
        "endpoint": _url,
        "version": 3,
        "table_name": _table_name,
        "fields": _player_box_score_skeleton(list(BOX_SCORE_STAT_MAPS[_key].keys())),
    }


# ── Dispatch ─────────────────────────────────────────────────────────────────

def fetch_endpoint(key: str, params: dict):
    """Generic dispatcher — the direct equivalent of the original func_run_proc()."""
    spec = ENDPOINTS[key]
    url_params = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{spec['endpoint']}{url_params}"
    headers = CORE_API_HEADERS if key == "GAME_DATA" else STATS_HEADERS
    data = _fetch_json(url, headers, timeout=spec.get("timeout", 20.0))

    if spec["version"] == 3:
        return _filter_data_to_skeleton(data[spec["table_name"]], spec["fields"])
    if spec["version"] == 2:
        return _filter_results_by_list(data["resultSets"], spec["fields"], spec["table_name"])
    if spec["version"] == 4:
        # "modules" is legitimately an empty list on any date with zero
        # scheduled games — not a rare edge case: off-days, All-Star break,
        # and the entire offseason (this bit Airflow the first time it ran,
        # in September, months before tip-off) all return this shape. Treat
        # it as "no games today," not an error.
        modules = data.get("modules") or []
        cards = modules[0].get("cards", []) if modules else []
        return _get_items_in_list_of_dicts(cards, spec["fields"], spec["rename_fields"])
    raise ValueError(f"Unknown endpoint version for {key!r}")


# ── High-level fetchers ──────────────────────────────────────────────────────

def derive_season_from_game_id(game_id: str) -> str | None:
    """
    NBA game IDs encode the season directly: characters [3:5] are the
    season's start year as YY, e.g. "0022400231" -> "24" -> "2024-25".
    Returns None if game_id doesn't match the expected shape.
    """
    if not game_id or len(game_id) < 5 or not game_id[3:5].isdigit():
        return None
    start_year = 2000 + int(game_id[3:5])
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def discover_games_for_date(game_date: date) -> list[str]:
    rows = fetch_endpoint("GAME_DATA", {"GameDate": game_date.isoformat(), "platform": "web"})
    return [r["GAME_ID"] for r in rows]


def discover_games_for_season(season: str) -> list[str]:
    """
    Backfill helper (not used by the daily DAG — see life_os_nba_ingest.py's
    module docstring on how range backfills are triggered). leaguegamefinder
    returns one row per TEAM per game, so this dedupes by GAME_ID.
    """
    rows = fetch_endpoint("GAMES", {"Season": season, "LeagueID": "00"})
    seen: dict[str, bool] = {}
    for r in rows:
        seen[r["GAME_ID"]] = True
    return list(seen.keys())


def fetch_game_summary(game_id: str) -> dict:
    """Normalized flat dict — ready to become an nba_games row."""
    raw = fetch_endpoint("BS_SUMMARY", {"GameID": game_id})
    game_date = None
    if raw.get("gameEt"):
        game_date = datetime.strptime(raw["gameEt"], "%Y-%m-%dT%H:%M:%SZ").date()
    home = raw.get("homeTeam") or {}
    away = raw.get("awayTeam") or {}
    arena = raw.get("arena") or {}
    return {
        "game_date": game_date,
        "game_status": raw.get("gameStatus"),
        "game_status_text": raw.get("gameStatusText"),
        "period": raw.get("period"),
        "duration": raw.get("duration"),
        "attendance": raw.get("attendance"),
        "game_label": raw.get("gameLabel"),
        "game_sub_label": raw.get("gameSubLabel"),
        "series_text": raw.get("seriesText"),
        "arena_name": arena.get("arenaName"),
        "arena_city": arena.get("arenaCity"),
        "arena_state": arena.get("arenaState"),
        "home_team_id": home.get("teamId"),
        "home_score": home.get("score"),
        "home_wins": home.get("teamWins"),
        "home_losses": home.get("teamLosses"),
        "away_team_id": away.get("teamId"),
        "away_score": away.get("score"),
        "away_wins": away.get("teamWins"),
        "away_losses": away.get("teamLosses"),
    }


def fetch_box_score(endpoint_key: str, game_id: str) -> dict:
    """Works for BS_TRAD..BS_DEF and BS_MATCH alike — flattening differs, fetching doesn't."""
    return fetch_endpoint(endpoint_key, {"GameID": game_id})


def fetch_play_by_play(game_id: str) -> dict:
    return fetch_endpoint("PBP", {"GameID": game_id, "StartPeriod": "1", "EndPeriod": "4"})


def fetch_all_players(only_current_season: bool = True) -> list[dict]:
    """
    Defaults to the current season's roster only (a few hundred rows),
    not the full multi-decade historical list (several thousand rows,
    including players retired since the 1940s) — the latter is what the
    original script requested unconditionally (IsOnlyCurrentSeason=0),
    and it reliably read-timed-out from our Airflow host in production
    even with a generous timeout, most likely NBA's servers being slow
    to assemble that much history rather than anything specific to us.
    v1's only use for this data is naming players in recent box
    scores/play-by-play, which the current-season roster fully covers.

    Pass only_current_season=False for a one-off full historical backfill
    (e.g. before backfilling old seasons' box scores) — expect it to be
    slow and possibly need retrying; it is not run on the weekly schedule.
    """
    rows = fetch_endpoint("PLAYERS", {
        "LeagueID": "00",
        "IsOnlyCurrentSeason": "0" if not only_current_season else "1",
    })
    out = []
    for r in rows:
        out.append({
            "person_id": r["PERSON_ID"],
            "full_name": r["DISPLAY_FIRST_LAST"],
            "team_id": r.get("TEAM_ID") or None,   # NBA uses 0 for no-team; normalize to NULL
            "roster_status": r.get("ROSTERSTATUS"),
            "from_year": r.get("FROM_YEAR"),
            "to_year": r.get("TO_YEAR"),
        })
    return out


# ── Flatten helpers (nested parsed dict -> flat DB-ready rows) ──────────────

def flatten_player_box_score(parsed: dict, stat_map: dict, game_id: str) -> list[dict]:
    """
    Generic for all nine "one row per player" box-score variants — pass the
    matching entry from BOX_SCORE_STAT_MAPS as stat_map.
    """
    rows = []
    for side in ("homeTeam", "awayTeam"):
        team = parsed.get(side) or {}
        team_id = team.get("teamId")
        for p in team.get("players") or []:
            if p.get("personId") is None:
                continue  # skip team-total pseudo-rows some endpoints include
            stats = p.get("statistics") or {}
            row = {"game_id": game_id, "team_id": team_id, "person_id": p.get("personId"),
                   "position": p.get("position")}
            for src_key, dst_col in stat_map.items():
                row[dst_col] = stats.get(src_key)
            rows.append(row)
    return rows


def flatten_matchup_box_score(parsed: dict, game_id: str) -> list[dict]:
    rows = []
    for side in ("homeTeam", "awayTeam"):
        team = parsed.get(side) or {}
        team_id = team.get("teamId")
        for defender in team.get("players") or []:
            if defender.get("personId") is None:
                continue
            for m in defender.get("matchups") or []:
                if m.get("personId") is None:
                    continue
                s = m.get("statistics") or {}
                rows.append({
                    "game_id": game_id, "team_id": team_id,
                    "defender_person_id": defender.get("personId"),
                    "defender_position": defender.get("position"),
                    "offensive_person_id": m.get("personId"),
                    "matchup_minutes": s.get("matchupMinutes"),
                    "matchup_minutes_sort": s.get("matchupMinutesSort"),
                    "partial_possessions": s.get("PartialPossessions"),
                    "pct_defender_total_time": s.get("percentageDefenderTotalTime"),
                    "pct_offensive_total_time": s.get("percentageOffensiveTotalTime"),
                    "pct_total_time_both_on": s.get("percentageTotalTimeBothOn"),
                    "switches_on": s.get("switchesOn"),
                    "player_points": s.get("playerPoints"),
                    "team_points": s.get("teamPoints"),
                    "matchup_assists": s.get("matchupAssists"),
                    "matchup_potential_assists": s.get("matchupPotentialAssists"),
                    "matchup_turnovers": s.get("matchupTurnovers"),
                    "matchup_blocks": s.get("matchupBlocks"),
                    "matchup_field_goals_made": s.get("matchupFieldGoalsMade"),
                    "matchup_field_goals_attempted": s.get("matchupFieldGoalsAttempted"),
                    "matchup_field_goals_percentage": s.get("matchupFieldGoalsPercentage"),
                    "matchup_three_pointers_made": s.get("matchupThreePointersMade"),
                    "matchup_three_pointers_attempted": s.get("matchupThreePointersAttempted"),
                    "matchup_three_pointers_percentage": s.get("matchupThreePointersPercentage"),
                    "help_blocks": s.get("helpBlocks"),
                    "help_field_goals_made": s.get("helpFieldGoalsMade"),
                    "help_field_goals_attempted": s.get("helpFieldGoalsAttempted"),
                    "help_field_goals_percentage": s.get("helpFieldGoalsPercentage"),
                    "matchup_free_throws_made": s.get("matchupFreeThrowsMade"),
                    "matchup_free_throws_attempted": s.get("matchupFreeThrowsAttempted"),
                    "shooting_fouls": s.get("shootingFouls"),
                })
    return rows


def flatten_play_by_play(parsed: dict, game_id: str) -> list[dict]:
    rows = []
    for a in parsed.get("actions") or []:
        is_fg = a.get("isFieldGoal")
        rows.append({
            "game_id": game_id,
            "action_number": a.get("actionNumber"),
            "action_id": a.get("actionId"),
            "period": a.get("period"),
            "clock": a.get("clock"),
            "team_id": a.get("teamId"),
            "team_tricode": a.get("teamTricode"),
            "person_id": a.get("personId"),
            "player_name": a.get("playerName"),
            "action_type": a.get("actionType"),
            "sub_type": a.get("subType"),
            "description": a.get("description"),
            "score_home": a.get("scoreHome"),
            "score_away": a.get("scoreAway"),
            "points_total": a.get("pointsTotal"),
            "shot_distance": a.get("shotDistance"),
            "shot_result": a.get("shotResult"),
            "shot_value": a.get("shotValue"),
            "is_field_goal": bool(is_fg) if is_fg is not None else None,
            "location": a.get("location"),
            "shot_x": a.get("xLegacy"),
            "shot_y": a.get("yLegacy"),
        })
    return rows
