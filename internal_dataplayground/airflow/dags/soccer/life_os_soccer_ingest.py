# airflow/dags/soccer/life_os_soccer_ingest.py
"""
Daily FIFA data ingest for the soccer domain (WO#34).

Pulls fixtures/results for every watched competition in
soccer_competitions, upserts them into soccer_matches, and backfills
match detail + play-by-play payloads (raw only — see
domains/soccer/models.py's module docstring) for matches that have gone
live or finished since the last run.

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

from dag_db import fetch_all, fetch_one, execute
from agents.soccer_agents import (
    fetch_matches, fetch_match_details, fetch_match_events, parse_match_summary,
)

log = logging.getLogger(__name__)

DAG_ID = "life_os_soccer_ingest"

# ── WATCH LIST SEED ───────────────────────────────────────────────────────────
# The project's tracked interests (per WO#34's own text) plus the World
# Cup, which the source script used as its own test case. Adding a
# competition later is a DB insert (see fetch_competitions() in
# soccer_agents.py for IDs), not a code change — this list only controls
# what gets seeded on a fresh install. backfill_from_date is a rough
# "start of the period worth having history for" per competition, not
# meant to be precise — it only matters for the very first run.
_SEED_COMPETITIONS = [
    {"fifa_competition_id": "17",         "name": "FIFA World Cup",                "backfill_from_date": "2022-01-01"},
    {"fifa_competition_id": "2000001032", "name": "UEFA Champions League",         "backfill_from_date": "2024-07-01"},
    {"fifa_competition_id": "2000000000", "name": "Barclays Premier League",       "backfill_from_date": "2024-07-01"},
    {"fifa_competition_id": "2000000078", "name": "Campeonato Brasileiro Série A", "backfill_from_date": "2024-01-01"},
    {"fifa_competition_id": "2000001035", "name": "Copa Libertadores",             "backfill_from_date": "2024-01-01"},
]

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


def seed_watched_competitions():
    """Idempotently ensures every _SEED_COMPETITIONS row exists. Safe to run daily."""
    for comp in _SEED_COMPETITIONS:
        existing = fetch_one(
            "SELECT id FROM soccer_competitions WHERE fifa_competition_id = %s",
            (comp["fifa_competition_id"],),
        )
        if existing:
            continue
        execute(
            "INSERT INTO soccer_competitions "
            "(fifa_competition_id, name, is_active, backfill_from_date) "
            "VALUES (%s, %s, 1, %s)",
            (comp["fifa_competition_id"], comp["name"], comp["backfill_from_date"]),
        )
        log.info("Seeded watched competition: %s (%s)", comp["name"], comp["fifa_competition_id"])


def _store_raw(endpoint: str, fifa_competition_id, fifa_match_id, payload):
    execute(
        "INSERT INTO soccer_raw_payloads (endpoint, fifa_competition_id, fifa_match_id, payload) "
        "VALUES (%s, %s, %s, %s)",
        (endpoint, fifa_competition_id, fifa_match_id, json.dumps(payload)),
    )


def _upsert_match(competition_row_id: int, parsed: dict):
    """Insert-or-update one match by its FIFA composite identity."""
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
            "home_team_score=%s, away_team_score=%s, kickoff_at=%s, "
            "fifa_match_status_code=%s, status_label=%s, venue_name=%s "
            "WHERE id=%s",
            (parsed["home_team_name"], parsed["away_team_name"],
             parsed["home_team_score"], parsed["away_team_score"], kickoff_at,
             parsed["fifa_match_status_code"], parsed["status_label"],
             parsed["venue_name"], existing["id"]),
        )
    else:
        execute(
            "INSERT INTO soccer_matches "
            "(competition_id, fifa_competition_id, fifa_season_id, fifa_stage_id, fifa_match_id, "
            "home_team_name, away_team_name, home_team_score, away_team_score, kickoff_at, "
            "fifa_match_status_code, status_label, venue_name) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (competition_row_id, parsed["fifa_competition_id"], parsed["fifa_season_id"],
             parsed["fifa_stage_id"], parsed["fifa_match_id"],
             parsed["home_team_name"], parsed["away_team_name"],
             parsed["home_team_score"], parsed["away_team_score"], kickoff_at,
             parsed["fifa_match_status_code"], parsed["status_label"], parsed["venue_name"]),
        )


# ── TASK 1 — FIXTURES / RESULTS ───────────────────────────────────────────────

def ingest_fixtures():
    """Pulls fixtures/results for every active watched competition."""
    seed_watched_competitions()

    competitions = fetch_all(
        "SELECT id, fifa_competition_id, name, backfill_from_date "
        "FROM soccer_competitions WHERE is_active = 1"
    )
    today = date.today()
    past_days, future_days = _get_window_days()

    for comp in competitions:
        has_matches = fetch_one(
            "SELECT id FROM soccer_matches WHERE competition_id = %s LIMIT 1", (comp["id"],)
        )
        if not has_matches and comp["backfill_from_date"]:
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

        for raw_match in raw_matches:
            parsed = parse_match_summary(raw_match)
            _upsert_match(comp["id"], parsed)

        log.info("Ingested %d matches for %s", len(raw_matches), comp["name"])


# ── TASK 2 — MATCH DETAILS + PLAY-BY-PLAY (raw only) ─────────────────────────

def ingest_match_details():
    """Backfills /live + /timelines (raw only) for live/finished matches not yet fetched."""
    pending = fetch_all(
        "SELECT id, fifa_competition_id, fifa_season_id, fifa_stage_id, fifa_match_id "
        "FROM soccer_matches WHERE status_label IN ('live', 'finished') AND details_fetched_at IS NULL"
    )

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

            execute(
                "UPDATE soccer_matches SET details_fetched_at = %s WHERE id = %s",
                (datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), match["id"]),
            )
        except Exception as exc:
            log.error("Detail/events fetch failed for match %s: %s", match["fifa_match_id"], exc)
            continue

    log.info("Processed detail/events backfill for %d candidate matches", len(pending))


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

    fixtures_task >> details_task
