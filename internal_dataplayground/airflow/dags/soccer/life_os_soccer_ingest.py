# airflow/dags/soccer/life_os_soccer_ingest.py
"""
Daily FIFA data ingest for the soccer domain (WO#34).

Four tasks, in order:
  1. ingest_fixtures            — pulls fixtures/results for every
                                   watched competition, upserts into
                                   soccer_matches.
  2. ingest_match_details       — fetches /live + /timelines (raw only)
                                   for matches worth refreshing. Makes
                                   FIFA API calls — bounded to the
                                   rolling ingestion window for cost
                                   reasons (see its own docstring).
  3. parse_finished_match_details — parses whatever raw payloads already
                                   exist into the normalized lineup/goal/
                                   booking/substitution/coach tables plus
                                   derived scalar stats. Makes NO API
                                   calls — pure local reprocessing, so it
                                   is NOT bounded to the window the same
                                   way (see its own docstring).
  4. prune_raw_payloads         — keeps soccer_raw_payloads from growing
                                   without bound, while still preserving
                                   a small amount of history for
                                   debugging (see its own docstring).

REDESIGNED 2026-09-18. The previous design fetched/parsed a match's
detail data ONCE and locked it forever once "finished". That failed in
a very concrete way this session: a real match's soccer_goals rows
looked like duplicates (4 goals for a 3-goal side) because the /live
snapshot had been captured while a penalty shootout was still in
progress — confirmed by the project owner, who watched the match. The
lock meant that incomplete snapshot was never going to be refreshed.
The new design accepts that ANY single snapshot might be incomplete and
just keeps refreshing anything within the rolling window until it ages
out, rather than trusting the first "finished" status it sees.
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
    parse_match_lineup_data, parse_match_event_stats,
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
#
# This SAME window now also governs how far back/forward
# ingest_match_details() reaches to refresh detail data — see that
# function's docstring.
ROLLING_WINDOW_PAST_DAYS = 3
ROLLING_WINDOW_FUTURE_DAYS = 60

# How far ahead ingest_match_details() reaches. Deliberately much shorter
# than the fixtures window: /live and /timelines for a match days away
# hold no lineup or events yet, so fetching them daily for every
# scheduled fixture in the next 60 days only burns API calls and adds
# useless raw rows. A 1-day lookahead catches lineups published before
# kickoff and matches kicking off late in the day. ingest_fixtures() and
# parse_finished_match_details() are NOT affected.
DETAIL_LOOKAHEAD_DAYS = 1


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
    matters: backfill_from_date is editable after the fact from
    /soccer/settings (e.g. "actually, pull World Cup history back to
    2018 too"), and the old rule would have silently ignored that edit
    forever once a competition already had at least one match. With this
    rule, editing it to an earlier date means the very next run notices
    the gap and pulls the wider range, then settles back to the cheap
    rolling window once the gap closes.

    Used only by ingest_fixtures() for the FIXTURES pull — unrelated to
    the match-detail reprocessing window used by the other two tasks.
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
    that column's meaning unambiguous. It's now purely informational
    (see SoccerMatch's docstring) rather than a processing gate, but
    it's still confusing to leave a stale timestamp sitting on a match
    that's gone back to 'scheduled' or similar.
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

    The watch list itself is NOT seeded or managed here — soccer_competitions
    is seeded once by the s0cc3r_d0ma1n001 migration on initial install,
    and from there is entirely self-service via /soccer/settings.
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
                log.error(
                    "Failed to parse/upsert one match for %s (raw IdMatch=%s): %s",
                    comp["name"], raw_match.get("IdMatch"), exc,
                )
                continue

        log.info("Ingested %d/%d matches for %s", upserted, len(raw_matches), comp["name"])


# ── TASK 2 — MATCH DETAILS + PLAY-BY-PLAY (raw fetch, API calls) ─────────────

def ingest_match_details():
    """
    Fetches /live + /timelines (raw only) and stores them in
    soccer_raw_payloads.

    REDESIGNED 2026-09-18: refetches EVERY match whose kickoff falls
    between the start of the rolling ingestion window (see
    _get_window_days()) and DETAIL_LOOKAHEAD_DAYS ahead (not the full
    future window — see that constant), unconditionally,
    on every run — not gated by status_label or a one-time lock. Outside
    that window, a match is only fetched if it has NEVER had a
    match_details payload at all — this still catches historical/
    backfilled matches that need at least one pull, without perpetually
    re-hitting FIFA for settled history forever.

    This is the bounded, API-calling half of the redesign — see
    parse_finished_match_details() for the unbounded, API-free half. The
    split exists because a stale/incomplete snapshot (the real case that
    motivated this: a penalty shootout still in progress when fetched)
    can only be fixed by a FRESH fetch, not by re-parsing the same old
    data — so re-fetching stays bounded to where it's actually likely to
    matter (recent activity), while re-parsing, being free, doesn't need
    to be.
    """
    past_days, future_days = _get_window_days()
    today = date.today()
    window_start = today - timedelta(days=past_days)
    # Upper bound is the short detail lookahead, capped by the configured
    # future window, taken to END of that day (a bare date would compare
    # as midnight and exclude that day's own kickoffs).
    lookahead_days = min(future_days, DETAIL_LOOKAHEAD_DAYS)
    window_end = datetime.combine(today + timedelta(days=lookahead_days), datetime.max.time())

    # Both branches are capped at window_end: the "never fetched" branch
    # still sweeps up old/backfilled matches, but no longer drags in every
    # future fixture. Matches with NULL kickoff_at are skipped (they have
    # no details to fetch yet).
    pending = fetch_all(
        """
        SELECT id, fifa_competition_id, fifa_season_id, fifa_stage_id, fifa_match_id
        FROM soccer_matches
        WHERE kickoff_at <= %s
          AND (
               kickoff_at >= %s
               OR NOT EXISTS (
                   SELECT 1 FROM soccer_raw_payloads
                   WHERE endpoint = 'match_details' AND fifa_match_id = soccer_matches.fifa_match_id
               )
          )
        """,
        (window_end, window_start),
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
        except Exception as exc:
            log.error("Detail/events fetch failed for match %s: %s", match["fifa_match_id"], exc)
            continue

    log.info(
        "Fetched details/events for %d/%d candidate matches (window: %s to %s)",
        fetched, len(pending), window_start, window_end,
    )


# ── TASK 3 — PARSE LINEUPS/GOALS/BOOKINGS/SUBS/COACHES/STATS (local, free) ───

def parse_finished_match_details():
    """
    Parses whatever raw /live + /timelines payloads already exist in
    soccer_raw_payloads into soccer_match_lineups / soccer_goals /
    soccer_bookings / soccer_substitutions / soccer_coaches, plus derived
    scalar fields on soccer_matches (formation, possession, attendance,
    penalty scores, shots/corners/fouls/offsides).

    REDESIGNED 2026-09-18 — now a FULL OVERWRITE, not insert-if-missing.
    Every match whose kickoff falls within the current rolling window
    gets its five normalized tables deleted and reinserted from the MOST
    RECENT raw payload on every run, even if it already had rows.
    Outside that window, a match is only (re)processed if it has never
    been parsed at all (no soccer_match_lineups rows yet).

    That second branch is what makes a full "clean slate" cheap: clear
    the five normalized tables, and on the very next run EVERY match —
    including years of backfilled history — becomes eligible for a fresh
    parse pass, entirely from data already sitting in
    soccer_raw_payloads. No new FIFA API calls needed for anything
    outside the rolling window; see ingest_match_details() for the half
    of this redesign that DOES call FIFA, and why that half stays
    bounded while this one doesn't.

    Safe to re-run if it ever partially fails: the DELETEs and INSERTs
    for one match all go through a single execute_many() call, which
    dag_db.py runs as one transaction, so a failure leaves that match's
    normalized rows exactly as they were before this attempt (old data,
    not half-deleted) — the guard picks it up again next run either way.
    """
    past_days, future_days = _get_window_days()
    today = date.today()
    window_start = today - timedelta(days=past_days)
    window_end = today + timedelta(days=future_days)

    pending = fetch_all(
        """
        SELECT id, fifa_competition_id, fifa_season_id, fifa_stage_id, fifa_match_id,
               fifa_home_team_id, fifa_away_team_id
        FROM soccer_matches
        WHERE kickoff_at BETWEEN %s AND %s
           OR NOT EXISTS (
               SELECT 1 FROM soccer_match_lineups WHERE match_id = soccer_matches.id
           )
        """,
        (window_start, window_end),
    )

    parsed_count = 0
    for match in pending:
        details_row = fetch_one(
            "SELECT payload FROM soccer_raw_payloads "
            "WHERE endpoint = 'match_details' AND fifa_match_id = %s "
            "ORDER BY fetched_at DESC LIMIT 1",
            (match["fifa_match_id"],),
        )
        if not details_row:
            # No match_details payload captured yet for this match at
            # all (ingest_match_details() hasn't run for it this cycle,
            # or ever) — nothing to parse yet.
            continue

        try:
            payload = details_row["payload"]
            raw_details = json.loads(payload) if isinstance(payload, str) else payload
            parsed = parse_match_lineup_data(raw_details)
        except Exception as exc:
            log.error("Failed to parse match_details for %s: %s", match["fifa_match_id"], exc)
            continue

        # match_events is optional — a details payload can exist without
        # a matching events payload (e.g. fetch_match_events() failed
        # independently of fetch_match_details() succeeding). Missing
        # event stats just come back NULL rather than blocking the rest
        # of this match's parse.
        event_stats = {
            "shots_home": None, "shots_away": None,
            "corners_home": None, "corners_away": None,
            "fouls_home": None, "fouls_away": None,
            "offsides_home": None, "offsides_away": None,
        }
        events_row = fetch_one(
            "SELECT payload FROM soccer_raw_payloads "
            "WHERE endpoint = 'match_events' AND fifa_match_id = %s "
            "ORDER BY fetched_at DESC LIMIT 1",
            (match["fifa_match_id"],),
        )
        if events_row:
            try:
                events_payload = events_row["payload"]
                raw_events = json.loads(events_payload) if isinstance(events_payload, str) else events_payload
                event_stats = parse_match_event_stats(
                    raw_events, match["fifa_home_team_id"], match["fifa_away_team_id"],
                )
            except Exception as exc:
                log.error("Failed to parse match_events for %s: %s", match["fifa_match_id"], exc)

        execute(
            "UPDATE soccer_matches SET home_formation=%s, away_formation=%s, "
            "possession_home=%s, possession_away=%s, attendance=%s, "
            "home_penalty_score=%s, away_penalty_score=%s, "
            "shots_home=%s, shots_away=%s, corners_home=%s, corners_away=%s, "
            "fouls_home=%s, fouls_away=%s, offsides_home=%s, offsides_away=%s, "
            "details_fetched_at=%s "
            "WHERE id=%s",
            (parsed["home_formation"], parsed["away_formation"],
             parsed["possession_home"], parsed["possession_away"], parsed["attendance"],
             parsed["home_penalty_score"], parsed["away_penalty_score"],
             event_stats["shots_home"], event_stats["shots_away"],
             event_stats["corners_home"], event_stats["corners_away"],
             event_stats["fouls_home"], event_stats["fouls_away"],
             event_stats["offsides_home"], event_stats["offsides_away"],
             datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), match["id"]),
        )

        # Full overwrite: clear whatever's there (possibly a stale/
        # incomplete snapshot from a previous run) before reinserting
        # fresh rows, all in one transaction.
        statements = [
            ("DELETE FROM soccer_match_lineups WHERE match_id = %s", (match["id"],)),
            ("DELETE FROM soccer_goals WHERE match_id = %s", (match["id"],)),
            ("DELETE FROM soccer_bookings WHERE match_id = %s", (match["id"],)),
            ("DELETE FROM soccer_substitutions WHERE match_id = %s", (match["id"],)),
            ("DELETE FROM soccer_coaches WHERE match_id = %s", (match["id"],)),
        ]
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

        execute_many(statements)
        parsed_count += 1

    log.info(
        "Reprocessed lineup/event/stat data for %d/%d candidate matches (window: %s to %s)",
        parsed_count, len(pending), window_start, window_end,
    )


# ── TASK 4 — PRUNE RAW PAYLOAD HISTORY (local, free) ──────────────────────────

RAW_PAYLOAD_RETENTION_COUNT = 3


def prune_raw_payloads():
    """
    Keeps only the RAW_PAYLOAD_RETENTION_COUNT most recent rows per
    (endpoint, fifa_competition_id, fifa_match_id) group in
    soccer_raw_payloads, deleting older snapshots.

    ADDED 2026-09-18: this table was growing without bound — every
    reprocessing of a match added a new row instead of replacing the old
    one, and the window-based full-overwrite reprocessing above makes
    that WORSE, not better, since matches inside the rolling window now
    get refetched on every run instead of once ever.

    This PRUNES, it does not collapse to a single row per group — losing
    all history would remove exactly what made two real issues
    diagnosable this session (a match_events payload confirmed going
    from empty to populated across separate pulls; a soccer_goals
    puzzle resolved by checking status_label against raw history).
    RAW_PAYLOAD_RETENTION_COUNT is deliberately small but not 1, as the
    balance between bounding growth and keeping that value.

    fifa_match_id is NULL for the "matches" (calendar page) endpoint —
    COALESCE'd to '' so those rows group correctly per competition
    instead of all colliding under one NULL bucket.
    """
    execute(
        """
        DELETE FROM soccer_raw_payloads
        WHERE id IN (
            SELECT id FROM (
                SELECT id, ROW_NUMBER() OVER (
                    PARTITION BY endpoint, fifa_competition_id, COALESCE(fifa_match_id, '')
                    ORDER BY fetched_at DESC
                ) AS rn
                FROM soccer_raw_payloads
            ) ranked
            WHERE rn > %s
        )
        """,
        (RAW_PAYLOAD_RETENTION_COUNT,),
    )
    log.info(
        "Pruned soccer_raw_payloads to the %d most recent snapshots per (endpoint, competition, match)",
        RAW_PAYLOAD_RETENTION_COUNT,
    )


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

    prune_task = PythonOperator(
        task_id="prune_raw_payloads",
        python_callable=prune_raw_payloads,
    )

    fixtures_task >> details_task >> lineups_task >> prune_task
