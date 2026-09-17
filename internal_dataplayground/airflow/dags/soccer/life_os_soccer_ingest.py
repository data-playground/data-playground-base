# airflow/dags/soccer/life_os_soccer_ingest.py
"""
Daily FIFA data ingest for the soccer domain (WO#34).

Pulls fixtures/results for every watched competition in
soccer_competitions, upserts them into soccer_matches, and backfills
match detail + play-by-play payloads (raw only — see
domains/soccer/models.py's module docstring) for matches near kickoff or
already finished — see ingest_match_details()'s own docstring for the
two different re-fetch policies and the 2026-09-12 fix to how "already
fetched" is determined.

Per CONTRIBUTING.md / GOVERNANCE.md §2.2: this DAG never imports
models.py, database.py, or any router/service. All DB access goes
through dag_db.py's raw-SQL helpers. All FIFA HTTP calls go through
airflow.agents.soccer_agents, which is likewise DAG-safe (no ORM, no
FastAPI imports).
"""
import sys
import json
import logging
from datetime import datetime, timedelta, date

sys.path.insert(0, '/opt/airflow/project')
sys.path.insert(0, '/opt/airflow/project/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator

from dag_db import fetch_all, fetch_one, execute, execute_many
from agents.soccer_agents import (
    fetch_matches, fetch_match_details, fetch_match_events, parse_match_summary,
    parse_match_lineup_data,
)

log = logging.getLogger(__name__)

DAG_ID = "life_os_soccer_ingest"

# Rolling window applied on every run after a competition's first
# (keeps the daily job cheap — recent-result corrections + near-term
# schedule, not a full re-pull of history every day). These are only the
# FALLBACK values now — see _get_window_days() below. They're also the
# values the soccer_settings row is seeded with (both in the WO#34
# migration and in SoccerSettings' model-level column defaults), so a
# fresh install behaves identically whether or not that row exists yet.
ROLLING_WINDOW_PAST_DAYS = 3
ROLLING_WINDOW_FUTURE_DAYS = 60


def _get_window_days() -> tuple[int, int]:
    """
    Reads the user-tunable window from soccer_settings (edited from
    /soccer/settings — see domains/soccer/routers/soccer_settings.py).
    Falls back to the module constants above if the table is empty, so
    this DAG never hard-fails just because the FastAPI app hasn't created
    the settings row yet (e.g. a brand-new install where the DAG's first
    scheduled run happens to fire before anyone has loaded the settings
    page even once).
    """
    row = fetch_one("SELECT window_past_days, window_future_days FROM soccer_settings LIMIT 1")
    if row:
        return row["window_past_days"], row["window_future_days"]
    return ROLLING_WINDOW_PAST_DAYS, ROLLING_WINDOW_FUTURE_DAYS


def _needs_backfill(comp: dict) -> bool:
    """
    True if this competition's configured backfill_from_date reaches
    further back than the earliest match currently on file for it — i.e.
    there's a gap to fill.

    This is re-checked on EVERY run, not just "does this competition have
    zero matches yet" (the original, simpler rule). That distinction
    matters: backfill_from_date is now editable after the fact from
    /soccer/settings (e.g. "actually, pull World Cup history back to
    2018 too"), and the old rule would have silently ignored that edit
    forever once a competition already had at least one match. With this
    rule, editing it to an earlier date means the very next run notices
    the gap and pulls the wider range — a one-time larger fetch until the
    gap closes, then it settles back to the cheap rolling window on
    subsequent runs since the earliest match on file will then already
    reach back to (or past) backfill_from_date.
    """
    backfill = comp["backfill_from_date"]
    if not backfill:
        return False
    earliest_row = fetch_one(
        "SELECT MIN(kickoff_at) AS earliest FROM soccer_matches WHERE competition_id = %s",
        (comp["id"],),
    )
    earliest = earliest_row["earliest"] if earliest_row else None
    if earliest is None:
        return True
    earliest_date = earliest.date() if hasattr(earliest, "date") else earliest
    return earliest_date > backfill


def _store_raw(endpoint: str, fifa_competition_id, fifa_match_id, payload):
    execute(
        "INSERT INTO soccer_raw_payloads (endpoint, fifa_competition_id, fifa_match_id, payload) "
        "VALUES (%s, %s, %s, %s)",
        (endpoint, fifa_competition_id, fifa_match_id, json.dumps(payload)),
    )


def _upsert_match(competition_row_id: int, parsed: dict):
    """
    Insert-or-update one match by its FIFA composite identity.

    On UPDATE, details_fetched_at is cleared back to NULL whenever the
    freshly-parsed status is anything other than 'finished' — keeping
    that column's meaning unambiguous ("non-null == this match is
    CURRENTLY finished and we've captured its final detail snapshot").
    Without this, a match that was ever incorrectly locked in as
    'finished' (as happened under the pre-2026-09-10 MatchStatus map bug
    — see ingest_match_details()'s docstring) would keep showing a stale,
    confusing details_fetched_at timestamp indefinitely even after
    status_label self-corrects back to 'scheduled'. It's harmless either
    way — ingest_match_details()'s query doesn't consult
    details_fetched_at at all for non-finished matches — but leaving it
    stale serves no purpose and is confusing to read directly.
    """
    if not parsed["fifa_match_id"]:
        return

    existing = fetch_one(
        "SELECT id FROM soccer_matches WHERE fifa_competition_id = %s AND fifa_season_id = %s "
        "AND fifa_stage_id = %s AND fifa_match_id = %s",
        (parsed["fifa_competition_id"], parsed["fifa_season_id"],
         parsed["fifa_stage_id"], parsed["fifa_match_id"]),
    )

    kickoff_at = None
    if parsed.get("kickoff_at"):
        try:
            kickoff_at = datetime.fromisoformat(parsed["kickoff_at"].replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            kickoff_at = None

    if existing:
        execute(
            "UPDATE soccer_matches SET home_team_name=%s, away_team_name=%s, "
            "fifa_home_team_id=%s, fifa_away_team_id=%s, "
            "home_team_score=%s, away_team_score=%s, kickoff_at=%s, "
            "fifa_match_status_code=%s, status_label=%s, venue_name=%s, "
            "details_fetched_at = CASE WHEN %s = 'finished' THEN details_fetched_at ELSE NULL END "
            "WHERE id=%s",
            (parsed["home_team_name"], parsed["away_team_name"],
             parsed["fifa_home_team_id"], parsed["fifa_away_team_id"],
             parsed["home_team_score"], parsed["away_team_score"], kickoff_at,
             parsed["fifa_match_status_code"], parsed["status_label"],
             parsed["venue_name"], parsed["status_label"], existing["id"]),
        )
    else:
        execute(
            "INSERT INTO soccer_matches "
            "(competition_id, fifa_competition_id, fifa_season_id, fifa_stage_id, fifa_match_id, "
            "home_team_name, away_team_name, fifa_home_team_id, fifa_away_team_id, "
            "home_team_score, away_team_score, kickoff_at, "
            "fifa_match_status_code, status_label, venue_name) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (competition_row_id, parsed["fifa_competition_id"], parsed["fifa_season_id"],
             parsed["fifa_stage_id"], parsed["fifa_match_id"],
             parsed["home_team_name"], parsed["away_team_name"],
             parsed["fifa_home_team_id"], parsed["fifa_away_team_id"],
             parsed["home_team_score"], parsed["away_team_score"], kickoff_at,
             parsed["fifa_match_status_code"], parsed["status_label"], parsed["venue_name"]),
        )


# ── TASK 1 — FIXTURES / RESULTS ───────────────────────────────────────────────

def ingest_fixtures():
    """
    Pulls fixtures/results for every active watched competition.

    The watch list itself is NOT seeded or managed here anymore — it
    used to be (see git history / the WO#34 conversation for why that
    was removed). soccer_competitions is seeded once by the
    s0cc3r_d0ma1n001 migration on initial install, and from there is
    entirely self-service via /soccer/settings (search FIFA's live
    competitions list and add/remove/deactivate). This DAG's only job is
    to iterate over whatever's active in that table — it has no opinion
    about which competitions "should" be watched.
    """
    competitions = fetch_all(
        "SELECT id, fifa_competition_id, name, backfill_from_date "
        "FROM soccer_competitions WHERE is_active = 1"
    )
    today = date.today()
    past_days, future_days = _get_window_days()
    log.info(
        "Ingest window this run: %d days past, %d days future "
        "(from soccer_settings if that row exists, else the module fallback constants)",
        past_days, future_days,
    )

    for comp in competitions:
        if _needs_backfill(comp):
            backfill = comp["backfill_from_date"]
            from_date = backfill.isoformat() if hasattr(backfill, "isoformat") else str(backfill)[:10]
        else:
            from_date = (today - timedelta(days=past_days)).isoformat()
        to_date = (today + timedelta(days=future_days)).isoformat()

        try:
            raw_matches = fetch_matches(comp["fifa_competition_id"], from_date, to_date)
        except Exception as exc:
            log.error("Fixture fetch failed for %s (%s): %s", comp["name"], comp["fifa_competition_id"], exc)
            continue

        _store_raw("matches", comp["fifa_competition_id"], None, raw_matches)

        upserted = 0
        for raw_match in raw_matches:
            try:
                parsed = parse_match_summary(raw_match)
                _upsert_match(comp["id"], parsed)
                upserted += 1
            except Exception as exc:
                # A single malformed match (see soccer_agents.py's
                # _extract_localized_text() docstring for the incident
                # that motivated this) used to kill the whole task,
                # discarding every other match already fetched this run.
                # The raw payload for ALL matches was already committed
                # above regardless, so nothing is lost either way — this
                # just stops one bad row from blocking the other ~167.
                log.error(
                    "Failed to parse/upsert one match for %s (raw IdMatch=%s): %s",
                    comp["name"], raw_match.get("IdMatch"), exc,
                )
                continue

        log.info("Ingested %d/%d matches for %s", upserted, len(raw_matches), comp["name"])


# ── TASK 2 — MATCH DETAILS + PLAY-BY-PLAY (raw only) ─────────────────────────

# How close to kickoff a non-finished match has to be before it's worth
# re-checking /live and /timelines on every run. Lineups get announced
# shortly before kickoff and events happen during the match, so this data
# is only worth repeatedly polling right around matchday — not for a
# fixture three months out. Not yet exposed on /soccer/settings (unlike
# the fixtures window), but could be if it turns out to need tuning.
DETAIL_FETCH_PROXIMITY_DAYS = 1


def ingest_match_details():
    """
    Backfills /live + /timelines (raw only).

    Two different re-fetch policies depending on match state:
      - FINISHED matches are fetched exactly once (details_fetched_at
        gates this) — the result is final and won't change.
      - Anything else (scheduled, postponed, or a not-yet-confirmed
        "live" state) is refetched on EVERY run, but only within
        +/- DETAIL_FETCH_PROXIMITY_DAYS of kickoff. details_fetched_at is
        deliberately left NULL for these fetches — only a fetch that
        lands on a genuinely 'finished' match locks it in for good.

    CONTEXT (2026-09-12): this replaces the original rule of
    "status_label IN ('live','finished') AND details_fetched_at IS NULL".
    Under the ORIGINAL (buggy) MatchStatus map, MatchStatus code 1 —
    which actually means "Scheduled" — was mistakenly mapped to "live".
    Any still-scheduled match sitting at that code got a premature /live
    fetch, which correctly came back with empty Players/Goals/Bookings/
    Substitutions (the match hadn't started — FIFA had nothing to report
    yet), and then details_fetched_at got permanently set, freezing that
    match on an empty snapshot forever even after the status map was
    corrected. The new rule fixes this going forward AND self-heals it:
    since details_fetched_at is now only trusted for matches that are
    CURRENTLY 'finished', any match wrongly locked in while still
    scheduled becomes eligible for refetch again as soon as it's within
    the proximity window, regardless of what details_fetched_at was set
    to under the old logic.
    """
    pending = fetch_all(
        """
        SELECT id, fifa_competition_id, fifa_season_id, fifa_stage_id, fifa_match_id, status_label
        FROM soccer_matches
        WHERE (status_label = 'finished' AND details_fetched_at IS NULL)
           OR (status_label != 'finished'
               AND kickoff_at BETWEEN (NOW() - INTERVAL %s DAY) AND (NOW() + INTERVAL %s DAY))
        """,
        (DETAIL_FETCH_PROXIMITY_DAYS, DETAIL_FETCH_PROXIMITY_DAYS),
    )

    fetched = 0
    for match in pending:
        try:
            details = fetch_match_details(
                match["fifa_competition_id"], match["fifa_season_id"],
                match["fifa_stage_id"], match["fifa_match_id"],
            )
            _store_raw("match_details", match["fifa_competition_id"], match["fifa_match_id"], details)

            events = fetch_match_events(
                match["fifa_competition_id"], match["fifa_season_id"],
                match["fifa_stage_id"], match["fifa_match_id"],
            )
            _store_raw("match_events", match["fifa_competition_id"], match["fifa_match_id"], events)
            fetched += 1

            # Only lock this match in as "done forever" once it's
            # CURRENTLY finished (per soccer_matches.status_label, freshly
            # updated by ingest_fixtures earlier in this same DAG run —
            # fixtures_task runs before details_task). Anything else stays
            # eligible for refetch every run within the proximity window.
            if match["status_label"] == "finished":
                execute(
                    "UPDATE soccer_matches SET details_fetched_at = %s WHERE id = %s",
                    (datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), match["id"]),
                )
        except Exception as exc:
            log.error("Detail/events fetch failed for match %s: %s", match["fifa_match_id"], exc)
            continue

    log.info("Fetched details/events for %d/%d candidate matches", fetched, len(pending))


# ── TASK 3 — PARSE LINEUPS/GOALS/BOOKINGS/SUBS/COACHES (idempotent) ──────────

def parse_finished_match_details():
    """
    Parses the raw /live payload already sitting in soccer_raw_payloads
    into the normalized soccer_match_lineups / soccer_goals /
    soccer_bookings / soccer_substitutions / soccer_coaches tables, for
    every finished match that doesn't have lineup rows yet.

    Deliberately makes NO FIFA API calls — it only reads what
    ingest_match_details() already stored. This is what makes it both
    the ongoing ingest step AND the backfill mechanism for every match
    that was fetched before this parser existed: the NOT EXISTS guard
    below doesn't care when a match's raw payload was captured, only
    whether it's been parsed yet. No separate one-off backfill script is
    needed — this task catches everything, forever, just by running
    daily alongside the other two.

    Guarded by soccer_match_lineups specifically (not a separate
    "parsed" flag column) — and safe to re-run if it ever partially
    fails, since all the INSERTs for one match go through a single
    execute_many() call, which dag_db.py runs as one transaction. A
    partial failure leaves soccer_match_lineups empty for that match, so
    the NOT EXISTS guard picks it up again next run rather than treating
    it as done.
    """
    pending = fetch_all(
        """
        SELECT sm.id, sm.fifa_match_id
        FROM soccer_matches sm
        WHERE sm.status_label = 'finished'
          AND NOT EXISTS (SELECT 1 FROM soccer_match_lineups WHERE match_id = sm.id)
        """
    )

    parsed_count = 0
    for match in pending:
        raw_row = fetch_one(
            "SELECT payload FROM soccer_raw_payloads "
            "WHERE endpoint = 'match_details' AND fifa_match_id = %s "
            "ORDER BY fetched_at DESC LIMIT 1",
            (match["fifa_match_id"],),
        )
        if not raw_row:
            # Fetched-but-not-yet-detailed edge case: status is
            # 'finished' but ingest_match_details() hasn't run for it
            # yet this cycle (or ever). Nothing to parse yet — it'll be
            # picked up once a match_details row exists.
            continue

        try:
            payload = raw_row["payload"]
            raw_details = json.loads(payload) if isinstance(payload, str) else payload
            parsed = parse_match_lineup_data(raw_details)
        except Exception as exc:
            log.error("Failed to parse match_details for %s: %s", match["fifa_match_id"], exc)
            continue

        execute(
            "UPDATE soccer_matches SET home_formation=%s, away_formation=%s, "
            "possession_home=%s, possession_away=%s, attendance=%s WHERE id=%s",
            (parsed["home_formation"], parsed["away_formation"],
             parsed["possession_home"], parsed["possession_away"],
             parsed["attendance"], match["id"]),
        )

        statements = []
        for row in parsed["lineups"]:
            statements.append((
                "INSERT INTO soccer_match_lineups "
                "(match_id, team_side, fifa_player_id, shirt_number, player_name, "
                "position_code, is_starter, is_captain, on_field_at_finish) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (match["id"], row["team_side"], row["fifa_player_id"], row["shirt_number"],
                 row["player_name"], row["position_code"], row["is_starter"],
                 row["is_captain"], row["on_field_at_finish"]),
            ))
        for row in parsed["goals"]:
            statements.append((
                "INSERT INTO soccer_goals "
                "(match_id, team_side, fifa_player_id, fifa_assist_player_id, "
                "minute_display, minute_numeric, period) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (match["id"], row["team_side"], row["fifa_player_id"], row["fifa_assist_player_id"],
                 row["minute_display"], row["minute_numeric"], row["period"]),
            ))
        for row in parsed["bookings"]:
            statements.append((
                "INSERT INTO soccer_bookings "
                "(match_id, team_side, fifa_player_id, card_type_code, "
                "minute_display, minute_numeric, period, reason) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (match["id"], row["team_side"], row["fifa_player_id"], row["card_type_code"],
                 row["minute_display"], row["minute_numeric"], row["period"], row["reason"]),
            ))
        for row in parsed["substitutions"]:
            statements.append((
                "INSERT INTO soccer_substitutions "
                "(match_id, team_side, fifa_player_off_id, fifa_player_on_id, "
                "player_off_name, player_on_name, minute_display, minute_numeric, period) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (match["id"], row["team_side"], row["fifa_player_off_id"], row["fifa_player_on_id"],
                 row["player_off_name"], row["player_on_name"], row["minute_display"],
                 row["minute_numeric"], row["period"]),
            ))
        for row in parsed["coaches"]:
            statements.append((
                "INSERT INTO soccer_coaches (match_id, team_side, fifa_coach_id, name, role_code) "
                "VALUES (%s, %s, %s, %s, %s)",
                (match["id"], row["team_side"], row["fifa_coach_id"], row["name"], row["role_code"]),
            ))

        if statements:
            execute_many(statements)

        parsed_count += 1

    log.info("Parsed lineup/event data for %d/%d finished matches", parsed_count, len(pending))


# ── DAG DEFINITION ─────────────────────────────────────────────────────────────

default_args = {
    "owner": "life_os",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Daily FIFA fixtures/results ingest for watched competitions",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["soccer", "ingest"],
) as dag:

    fixtures_task = PythonOperator(
        task_id="ingest_fixtures",
        python_callable=ingest_fixtures,
    )

    details_task = PythonOperator(
        task_id="ingest_match_details",
        python_callable=ingest_match_details,
    )

    lineups_task = PythonOperator(
        task_id="parse_finished_match_details",
        python_callable=parse_finished_match_details,
    )

    fixtures_task >> details_task >> lineups_task
