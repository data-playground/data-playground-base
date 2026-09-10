# domains/nba/seed_data.py
"""
One-time seed script for the nba_teams reference table (WO#33).

Run manually, once, after the nba_teams table exists (i.e. after the
Alembic migration generated from domains/nba/models.py has been applied):

    python domains/nba/seed_data.py

Self-contained — does not import database.py, dag_db.py, or models.py.
This is a one-off ops script, not part of the app's request-time or DAG
import graph, so it just opens its own pymysql connection the same way
dag_db.py does (same host/user/database convention).

Team IDs/tricodes are stats.nba.com's own static team directory (unchanged
in years). Reconstructed here from public reference sources, not pulled
live from the API — spot-check a couple against a live boxscoresummary
pull before relying on this for real joins.
"""
import json
import os

import pymysql
import pymysql.cursors

# (team_id, tricode, full_name, conference, division)
TEAMS = [
    (1610612737, "ATL", "Atlanta Hawks", "East", "Southeast"),
    (1610612738, "BOS", "Boston Celtics", "East", "Atlantic"),
    (1610612751, "BKN", "Brooklyn Nets", "East", "Atlantic"),
    (1610612766, "CHA", "Charlotte Hornets", "East", "Southeast"),
    (1610612741, "CHI", "Chicago Bulls", "East", "Central"),
    (1610612739, "CLE", "Cleveland Cavaliers", "East", "Central"),
    (1610612742, "DAL", "Dallas Mavericks", "West", "Southwest"),
    (1610612743, "DEN", "Denver Nuggets", "West", "Northwest"),
    (1610612765, "DET", "Detroit Pistons", "East", "Central"),
    (1610612744, "GSW", "Golden State Warriors", "West", "Pacific"),
    (1610612745, "HOU", "Houston Rockets", "West", "Southwest"),
    (1610612754, "IND", "Indiana Pacers", "East", "Central"),
    (1610612746, "LAC", "LA Clippers", "West", "Pacific"),
    (1610612747, "LAL", "Los Angeles Lakers", "West", "Pacific"),
    (1610612763, "MEM", "Memphis Grizzlies", "West", "Southwest"),
    (1610612748, "MIA", "Miami Heat", "East", "Southeast"),
    (1610612749, "MIL", "Milwaukee Bucks", "East", "Central"),
    (1610612750, "MIN", "Minnesota Timberwolves", "West", "Northwest"),
    (1610612740, "NOP", "New Orleans Pelicans", "West", "Southwest"),
    (1610612752, "NYK", "New York Knicks", "East", "Atlantic"),
    (1610612760, "OKC", "Oklahoma City Thunder", "West", "Northwest"),
    (1610612753, "ORL", "Orlando Magic", "East", "Southeast"),
    (1610612755, "PHI", "Philadelphia 76ers", "East", "Atlantic"),
    (1610612756, "PHX", "Phoenix Suns", "West", "Pacific"),
    (1610612757, "POR", "Portland Trail Blazers", "West", "Northwest"),
    (1610612758, "SAC", "Sacramento Kings", "West", "Pacific"),
    (1610612759, "SAS", "San Antonio Spurs", "West", "Southwest"),
    (1610612761, "TOR", "Toronto Raptors", "East", "Atlantic"),
    (1610612762, "UTA", "Utah Jazz", "West", "Northwest"),
    (1610612764, "WAS", "Washington Wizards", "East", "Southeast"),
]


def _connect():
    mdb_json = json.loads(os.environ["MARIA_DB"])
    return pymysql.connect(
        host="db", user="data_playground", password=mdb_json["password"],
        database="jobs", cursorclass=pymysql.cursors.DictCursor, autocommit=False,
    )


def main() -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            for team_id, tricode, full_name, conf, div in TEAMS:
                cur.execute(
                    "INSERT INTO nba_teams (id, tricode, full_name, conference, division) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE tricode=VALUES(tricode), full_name=VALUES(full_name), "
                    "conference=VALUES(conference), division=VALUES(division)",
                    (team_id, tricode, full_name, conf, div),
                )
        conn.commit()
        print(f"Seeded {len(TEAMS)} teams.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
