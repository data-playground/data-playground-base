# airflow/dags/nba/life_os_nba_player_sync.py
"""
Weekly sync of the NBA player roster (WO#33).

STATUS — no longer essential. This DAG calls stats.nba.com's
commonallplayers, which times out from our deployment even after shrinking
the request to the current season only (the whole stats.nba.com host is
blocking us — see the WO#33 conversation history). Real player names now
reach nba_players a different way: life_os_nba_ingest.py's
_upsert_player_names() writes them straight from the box score rosters it
already fetches from core-api.nba.com (every rostered player appears there,
including bench, DNPs and inactives, each with a real name). What this DAG
would still add is career metadata the box score doesn't carry
(roster_status, from_year, to_year) and players who haven't appeared in
any box score yet. It's left in place, not deleted, in case stats.nba.com
access is restored or a core-api.nba.com roster endpoint is found — but
nothing in the app depends on it succeeding. Safe to pause in the Airflow UI
to stop the failing runs.


Runs on its own weekly schedule, separate from the nightly game-ingestion
DAG — see domains/nba/models.py's Player docstring for why: this is a
full-league roster snapshot (PLAYERS/commonallplayers), not per-game data,
so a nightly run would just be redundant load against stats.nba.com for
data that barely changes day to day.

Same DAG/FastAPI boundary rule as life_os_nba_ingest.py applies here: no
imports of models.py/database.py/routers/services — only dag_db.py's raw
SQL helpers and airflow/agents/nba_agents.py's pure fetch/parse functions.
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

from agents.nba_agents import fetch_all_players

log = logging.getLogger(__name__)


def task_sync_players() -> None:
    players = fetch_all_players()
    if not players:
        log.warning("PLAYERS endpoint returned no rows — skipping sync")
        return

    cols = ["person_id", "full_name", "team_id", "roster_status", "from_year", "to_year"]
    placeholders = ", ".join(["%s"] * len(cols))
    update_clause = ", ".join(f"{c}=VALUES({c})" for c in cols if c != "person_id")
    sql = (
        f"INSERT INTO nba_players ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_clause}"
    )
    statements = [(sql, tuple(p[c] for c in cols)) for p in players]
    execute_many(statements)
    log.info("Synced %d player(s)", len(players))


default_args = {
    "owner": "life-os",
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=5),
}

with DAG(
    dag_id="life_os_nba_player_sync",
    default_args=default_args,
    schedule_interval="0 6 * * 1",  # Mondays, 6am UTC
    start_date=datetime.datetime(2024, 10, 1),
    catchup=False,
    tags=["nba", "ingestion"],
) as dag:
    PythonOperator(
        task_id="sync_players",
        python_callable=task_sync_players,
    )
