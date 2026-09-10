# airflow/dags/nba/life_os_nba_ingest.py
"""
Daily ingestion DAG for the NBA domain (WO#33).

Discovers a day's games (defaults to "yesterday" — replaying
nba_game_summary.py's own default — or a specific date passed via
dag_run.conf), then for every discovered game pulls:
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

Backfill: this task processes one date per run. To backfill a range,
trigger this DAG once per date with `{"game_date": "YYYY-MM-DD"}` in the
run's conf — either from the Airflow UI's "Trigger DAG w/ config", or by
POSTing to the same dagRuns endpoint services/airflow_service.py already
wraps, once per date. A loop that fires many runs in a tight burst risks
the same rate-limiting nba_agents.py's throttle exists to avoid — space
backfill runs out (e.g. a few seconds apart) rather than firing them all
at once.
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
    fetch_game_summary,
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


def _target_date(conf: dict) -> datetime.date:
    if conf.get("game_date"):
        return datetime.date.fromisoformat(conf["game_date"])
    return datetime.date.today() - datetime.timedelta(days=1)


def task_ingest_games(**context) -> None:
    conf = (context.get("dag_run").conf if context.get("dag_run") else None) or {}
    target_date = _target_date(conf)

    game_ids = discover_games_for_date(target_date)
    log.info("Discovered %d game(s) for %s", len(game_ids), target_date)
    if not game_ids:
        return

    game_rows = []
    box_rows_by_key: dict[str, list[dict]] = {k: [] for k in BOX_SCORE_TABLES}
    matchup_rows: list[dict] = []
    pbp_rows: list[dict] = []

    for game_id in game_ids:
        summary = fetch_game_summary(game_id)

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

    log.info("Ingest complete for %s: %d game(s)", target_date, len(game_ids))


default_args = {
    "owner": "life-os",
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}

with DAG(
    dag_id="life_os_nba_ingest",
    default_args=default_args,
    schedule_interval="0 9 * * *",  # 9am UTC — after all US games have finished
    start_date=datetime.datetime(2024, 10, 1),
    catchup=False,
    tags=["nba", "ingestion"],
) as dag:
    PythonOperator(
        task_id="ingest_games",
        python_callable=task_ingest_games,
    )
