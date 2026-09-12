# airflow/dags/nba/life_os_nba_ingest.py
"""
Daily ingestion DAG for the NBA domain (WO#33).

Discovers a day's games, then for every discovered game pulls:
  - box score summary (game metadata: scores, arena, attendance, status)
  - all ten box score variants — traditional gets a dedicated table with a
    v1 UI; the other nine are ingested into normalized tables too, but are
    "headless" in this pass (browsable via /explorer, no dedicated page
    yet — see domains/nba/models.py's module docstring)
  - play-by-play

Per CONTRIBUTING.md / GOVERNANCE.md §2.2, this file never imports
models.py, database.py, or any router/service — all database access goes
through airflow/dag_db.py's raw-SQL helpers. All NBA-API-shape knowledge
(fetching + parsing + flattening) lives in airflow/agents/nba_agents.py,
which has no DB code of its own; this DAG is the only thing that writes
what nba_agents.py returns into MariaDB.

Which date(s) a run processes — three ways, all going through the same
_ingest_date() per date, so there's exactly one code path regardless of
how it was triggered:

  1. Normal daily schedule: no conf, so _resolve_dates() falls back to
     this run's own Airflow logical date (context["ds"]) — for a run
     actually executing on day D at 9am, that's D-1, replaying
     nba_game_summary.py's own "yesterday" default. This is also exactly
     what Airflow's own `airflow dags backfill -s START -e END
     life_os_nba_ingest` command relies on, so that command works
     out of the box with zero extra code here.

  2. dag_run.conf = {"game_date": "YYYY-MM-DD"} — single-date override,
     e.g. from the Airflow UI's "Trigger DAG w/ config".

  3. dag_run.conf = {"start_date": "...", "end_date": "..."} — a range,
     processed as a sequential loop *within this one task run*, one date
     at a time, upserting after each date (not batched across the whole
     range) so a multi-month backfill makes durable progress the whole
     way through rather than risking it all on one final write. Re-running
     is always safe — every write is an idempotent upsert (see _upsert()).

     A large range is a genuinely long-running task: ~12 HTTP requests per
     game (summary + 9 box-score variants + matchup + play-by-play) against
     nba_agents.py's throttle means roughly 8-10 seconds per game, so a
     full season (~1,200+ games) is realistically several hours. Consider
     backfilling a month or two at a time instead of an entire season in
     one run, both for visibility into progress and to keep a single task
     from tying up a worker slot for that long.

  4. Direct CLI, bypassing Airflow entirely — useful for watching a
     backfill's output live, or if you'd rather not build a conf JSON blob:

         python airflow/dags/nba/life_os_nba_ingest.py --date 2025-10-01
         python airflow/dags/nba/life_os_nba_ingest.py \
             --start-date 2025-10-01 --end-date 2025-10-31

     Same _ingest_date() call the DAG task makes — no separate code path
     to drift out of sync. Run it from wherever `agents` and `dag_db` are
     already importable (e.g. inside the Airflow container, same as the
     sys.path setup below assumes).

No settings page for this — deliberately. A one-time historical backfill
isn't a recurring workflow the way, say, finance CSV import settings are;
the Airflow UI's own "Trigger DAG w/ config" already covers option 2/3
above with no new code, and the CLI covers the same ground for anyone who'd
rather use a terminal. Revisit if this turns out to be something you
reach for often rather than a handful of times while backfilling history —
services/airflow_service.py's trigger_airflow(dag_id, conf) already exists
if a future settings page ever wants to fire this from a button.
"""
from __future__ import annotations

import datetime
import logging
import sys

sys.path.insert(0, "/opt/airflow/project")
sys.path.insert(0, "/opt/airflow/project/airflow")

from airflow import DAG
from airflow.operators.python import PythonOperator

from dag_db import execute_many

from agents.nba_agents import (
    BOX_SCORE_STAT_MAPS,
    derive_season_from_game_id,
    discover_games_for_date,
    fetch_box_score,
    fetch_game_details_header,
    fetch_play_by_play,
    flatten_matchup_box_score,
    flatten_play_by_play,
    flatten_player_box_score,
)

log = logging.getLogger(__name__)

# Endpoint key -> target table. BS_MATCH is handled separately below (its
# flatten shape doesn't fit the generic per-player pattern the other nine
# share — see nba_agents.flatten_matchup_box_score).
BOX_SCORE_TABLES = {
    "BS_TRAD":   "nba_box_score_traditional",
    "BS_ADV":    "nba_box_score_advanced",
    "BS_MISC":   "nba_box_score_misc",
    "BS_SCORE":  "nba_box_score_scoring",
    "BS_USAGE":  "nba_box_score_usage",
    "BS_FOUR":   "nba_box_score_fourfactors",
    "BS_TRACK":  "nba_box_score_tracking",
    "BS_HUSTLE": "nba_box_score_hustle",
    "BS_DEF":    "nba_box_score_defensive",
}


def _upsert(table: str, rows: list[dict], key_cols: list[str]) -> None:
    """
    Generic upsert — builds one parameterized INSERT ... ON DUPLICATE KEY
    UPDATE per row (from that row's own dict keys) and commits them all in
    one transaction via dag_db.execute_many(). Every row in a single call
    must share the same key set, which is true for every call site below
    (each comes from one endpoint's flatten_* function, which always
    returns uniform dicts). Relies on the UniqueConstraint each target
    table declares in domains/nba/models.py — that's what makes a re-run
    over the same date idempotent instead of duplicating rows.
    """
    if not rows:
        return
    cols = list(rows[0].keys())
    placeholders = ", ".join(["%s"] * len(cols))
    update_clause = ", ".join(f"{c}=VALUES({c})" for c in cols if c not in key_cols)
    sql = (
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_clause}"
    )
    statements = [(sql, tuple(row[c] for c in cols)) for row in rows]
    execute_many(statements)
    log.info("Upserted %d row(s) into %s", len(rows), table)


def _ensure_players_exist(person_ids: set[int]) -> None:
    """
    Box score / matchup / play-by-play rows all FK into nba_players, but
    that table is only filled by the *weekly* life_os_nba_player_sync.py —
    a player can debut (or get called up) between sync runs, and this
    nightly DAG must not fail on that. Inserts a minimal placeholder row
    for any person_id not already present, and does nothing if it already
    exists — "ON DUPLICATE KEY UPDATE person_id=person_id" is a genuine
    no-op, so a real name from a previous sync is never clobbered back to
    the placeholder. The next player-sync run overwrites the placeholder
    with the real name via its own upsert, same as any other player.
    """
    if not person_ids:
        return
    statements = [
        (
            "INSERT INTO nba_players (person_id, full_name) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE person_id = person_id",
            (pid, f"Player #{pid}"),
        )
        for pid in person_ids
    ]
    execute_many(statements)
    log.info("Ensured %d player row(s) exist (placeholder where not yet synced)", len(person_ids))


def _date_range(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    if end < start:
        raise ValueError(f"end date ({end}) is before start date ({start})")
    days = (end - start).days
    return [start + datetime.timedelta(days=i) for i in range(days + 1)]


def _resolve_dates(conf: dict, ds: str | None) -> list[datetime.date]:
    """
    Figures out which date(s) this run processes. Checked in order:
    explicit range, explicit single date, this run's own Airflow logical
    date (normal schedule + native `airflow dags backfill` both flow
    through here), then a bare today-1 fallback for the rare case this is
    invoked with no Airflow context at all.
    """
    if conf.get("start_date") and conf.get("end_date"):
        start = datetime.date.fromisoformat(conf["start_date"])
        end = datetime.date.fromisoformat(conf["end_date"])
        return _date_range(start, end)
    if conf.get("game_date"):
        return [datetime.date.fromisoformat(conf["game_date"])]
    if ds:
        return [datetime.date.fromisoformat(ds)]
    return [datetime.date.today() - datetime.timedelta(days=1)]


def _ingest_date(target_date: datetime.date, fetch_advanced_stats: bool = False) -> int:
    """
    Full ingest for exactly one date: discover -> fetch -> upsert. Returns
    the number of games ingested (0 on an off-day — not an error; see
    nba_agents.py's discover_games_for_date, which returns [] rather than
    raising when nothing's scheduled). The only function both the DAG task
    and the CLI entry point at the bottom of this file call — there is no
    second, parallel "backfill" implementation to drift out of sync.

    fetch_advanced_stats: when True, also fetches the 9 box-score variants,
    matchups, and play-by-play — all still sourced from stats.nba.com,
    which is currently blocking/timing out requests from this deployment
    (see the WO#33 conversation history — this isn't specific to our code).
    Defaults to False so a run — including a big backfill — populates
    nba_games (scores, status, records) quickly via
    fetch_game_details_header(), which uses core-api.nba.com and has
    stayed reachable throughout, rather than burning ~10 minutes per game
    on 11 calls known to fail. Flip back on once stats.nba.com access is
    restored (proxy, etc.) or those calls are replaced with an
    nba.com-scraping equivalent (in progress).
    """
    game_ids = discover_games_for_date(target_date)
    log.info("Discovered %d game(s) for %s", len(game_ids), target_date)
    if not game_ids:
        return 0

    game_rows = []
    box_rows_by_key: dict[str, list[dict]] = {k: [] for k in BOX_SCORE_TABLES}
    matchup_rows: list[dict] = []
    pbp_rows: list[dict] = []

    for game_id in game_ids:
        summary = fetch_game_details_header(game_id)

        game_rows.append({
            "game_id": game_id,
            "season": derive_season_from_game_id(game_id),
            "game_date": summary["game_date"] or target_date,
            "game_status": summary["game_status"],
            "game_status_text": summary["game_status_text"],
            "home_team_id": summary["home_team_id"],
            "away_team_id": summary["away_team_id"],
            "home_score": summary["home_score"],
            "away_score": summary["away_score"],
            "home_wins": summary["home_wins"],
            "home_losses": summary["home_losses"],
            "away_wins": summary["away_wins"],
            "away_losses": summary["away_losses"],
            "period": summary["period"],
            "duration": summary["duration"],
            "arena_name": summary["arena_name"],
            "arena_city": summary["arena_city"],
            "arena_state": summary["arena_state"],
            "attendance": summary["attendance"],
            "game_label": summary["game_label"],
            "game_sub_label": summary["game_sub_label"],
            "series_text": summary["series_text"],
        })

        if not fetch_advanced_stats:
            continue

        for endpoint_key in BOX_SCORE_TABLES:
            parsed = fetch_box_score(endpoint_key, game_id)
            box_rows_by_key[endpoint_key].extend(
                flatten_player_box_score(parsed, BOX_SCORE_STAT_MAPS[endpoint_key], game_id)
            )

        matchup_parsed = fetch_box_score("BS_MATCH", game_id)
        matchup_rows.extend(flatten_matchup_box_score(matchup_parsed, game_id))

        pbp_parsed = fetch_play_by_play(game_id)
        pbp_rows.extend(flatten_play_by_play(pbp_parsed, game_id))

    _upsert("nba_games", game_rows, key_cols=["game_id"])

    if fetch_advanced_stats:
        person_ids = {row["person_id"] for rows in box_rows_by_key.values() for row in rows if row.get("person_id")}
        person_ids |= {row["defender_person_id"] for row in matchup_rows if row.get("defender_person_id")}
        person_ids |= {row["offensive_person_id"] for row in matchup_rows if row.get("offensive_person_id")}
        person_ids |= {row["person_id"] for row in pbp_rows if row.get("person_id")}
        _ensure_players_exist(person_ids)

        for endpoint_key, table in BOX_SCORE_TABLES.items():
            _upsert(table, box_rows_by_key[endpoint_key], key_cols=["game_id", "person_id"])
        _upsert("nba_box_score_matchup", matchup_rows,
                key_cols=["game_id", "defender_person_id", "offensive_person_id"])
        _upsert("nba_play_by_play", pbp_rows, key_cols=["game_id", "action_number"])

    log.info("Ingest complete for %s: %d game(s) (advanced stats %s)",
              target_date, len(game_ids), "included" if fetch_advanced_stats else "skipped")
    return len(game_ids)


def task_ingest_games(**context) -> None:
    conf = (context.get("dag_run").conf if context.get("dag_run") else None) or {}
    dates = _resolve_dates(conf, context.get("ds"))
    fetch_advanced_stats = bool(conf.get("fetch_advanced_stats", False))

    if len(dates) > 1:
        log.info("Backfill range: %s date(s), %s .. %s", len(dates), dates[0], dates[-1])

    for i, d in enumerate(dates, start=1):
        if len(dates) > 1:
            log.info("[%d/%d] ingesting %s", i, len(dates), d)
        _ingest_date(d, fetch_advanced_stats=fetch_advanced_stats)


default_args = {
    "owner": "life-os",
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}


def _register_dag() -> DAG:
    """
    Builds the actual Airflow DAG object. Called only when this file is
    imported by Airflow's own DAG processor/scheduler (see the bottom of
    this file) — never when run directly as a CLI backfill script. A
    manual run has no reason to construct Airflow scaffolding at all, and
    doing so inside the same container the live scheduler runs in risks
    contending with it over this exact dag_id's metadata row, which is
    almost certainly why a direct `python3 life_os_nba_ingest.py` run can
    appear to hang before printing anything: it's stuck constructing a
    DAG object nothing is actually waiting on, not stuck on network I/O.
    """
    with DAG(
        dag_id="life_os_nba_ingest",
        default_args=default_args,
        schedule="0 9 * * *",  # 9am UTC — after all US games have finished. (Airflow 3 renamed schedule_interval -> schedule.)
        start_date=datetime.datetime(2024, 10, 1),
        catchup=False,
        tags=["nba", "ingestion"],
    ) as dag:
        PythonOperator(
            task_id="ingest_games",
            python_callable=task_ingest_games,
        )
    return dag


if __name__ != "__main__":
    dag = _register_dag()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Manually ingest NBA games for one date or a date range. "
                     "Same _ingest_date() the scheduled DAG task calls — safe "
                     "to re-run, every write is an idempotent upsert."
    )
    parser.add_argument("--date", help="Single date, YYYY-MM-DD.")
    parser.add_argument("--start-date", help="Range start (inclusive), YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Range end (inclusive), YYYY-MM-DD.")
    parser.add_argument(
        "--with-advanced-stats", action="store_true",
        help="Also fetch the 9 box-score variants, matchups, and play-by-play "
             "(stats.nba.com — currently blocked for us; off by default so a "
             "backfill isn't stuck burning ~10 min/game on calls known to fail).",
    )
    args = parser.parse_args()

    if args.date:
        cli_dates = [datetime.date.fromisoformat(args.date)]
    elif args.start_date and args.end_date:
        cli_dates = _date_range(
            datetime.date.fromisoformat(args.start_date),
            datetime.date.fromisoformat(args.end_date),
        )
    else:
        parser.error("Pass either --date, or both --start-date and --end-date.")

    print(f"Backfilling {len(cli_dates)} date(s): {cli_dates[0]} .. {cli_dates[-1]}")
    for idx, day in enumerate(cli_dates, start=1):
        print(f"[{idx}/{len(cli_dates)}] {day} ...", flush=True)
        n = _ingest_date(day, fetch_advanced_stats=args.with_advanced_stats)
        print(f"    {n} game(s) ingested" if n else "    no games scheduled")
    print("Done.")
