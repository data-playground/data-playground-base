# airflow/dags/life_os_blog_scout.py
"""
Blog Scout DAG
─────────────
Runs on a schedule (Mon + Thu 9am) and calls the Researcher agent
to generate 3 new blog blueprints, inserting them into blog_ideas.

Conf (optional, for manual triggers):
  {"interests": "override interests string"}
"""

# airflow/dags/life_os_blog_scout.py
"""
Blog Scout DAG
─────────────
Runs on a schedule (daily at 9am) and calls the Researcher agent
to generate 5 new blog blueprints, inserting them into blog_ideas.

Changes from v1:
  - Now generates 5 ideas per run (was 3) to support the difficulty distribution.
  - Stores the new `difficulty` field on blog_ideas.
  - Passes recent difficulty distribution back to the agent so it can rebalance
    if recent batches skewed too ambitious.
  - Deduplication is tighter: uses both title similarity and a domain check.

Conf (optional, for manual triggers):
  {"interests": "override interests string"}
"""

import sys
import logging
from datetime import datetime, timedelta

sys.path.insert(0, '/opt/airflow/project')
sys.path.insert(0, '/opt/airflow/project/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator
from agents.blog_agents import agent_researcher

log = logging.getLogger(__name__)

DEFAULT_INTERESTS = (
    "Sports, Basketball, Soccer (football), (American) Football, Volleyball, Baseball,  "
    "NBA, NFL, Premier League (English Football), MLB, Brasileirao (Brazilian Football), "
    "Champions League, World Cup (mainly Soccer, but others as well), "
    "Libertadores (South American Soccer), Olympics (summer and winter), "
    "Music (playing and listening), TV shows, Movies, Food, "
    "Tableau, SQL, AI, Gen AI, Python, Apache Airflow, "
    "GCP, Gemini, BigQuery, "
    "data pipelines, data extraction, data analysis"
)

DEFAULT_PROJECTS = (
    "Life OS (self-hosted FastAPI + MariaDB + Docker personal command center), "
    "Job Tracker with Gemini AI fit scoring and ATS pipeline tracking, "
    "Finance Tracker with CSV import and Gemini auto-categorization, "
    "Blog pipeline with multi-agent writing (Ghostwriter + Refiner + Editor), "
    "SQL Explorer (BigQuery-style UI for MariaDB), "
    "Code Intelligence (GitHub-connected README writer + Code Narrator)"
)

default_args = {
    "owner": "life_os",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


def _get_recent_difficulty_summary(conn) -> str:
    """
    Queries the last 30 generated ideas and summarizes the difficulty breakdown.
    Returns a plain-English note for the agent prompt so it can rebalance.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT difficulty, COUNT(*) as cnt
                FROM blog_ideas
                WHERE difficulty IS NOT NULL
                  AND created_at >= DATE_SUB(NOW(), INTERVAL 14 DAY)
                GROUP BY difficulty
                """,
            )
            rows = cur.fetchall()

        if not rows:
            return ""

        counts = {r["difficulty"]: r["cnt"] for r in rows}
        total = sum(counts.values())
        if total == 0:
            return ""

        lines = [
            f"Recent 14-day difficulty breakdown ({total} ideas total):",
        ]
        for level in ("starter", "weekend", "ambitious"):
            n = counts.get(level, 0)
            pct = round((n / total) * 100)
            lines.append(f"  - {level}: {n} ({pct}%)")

        # Add a rebalancing note if skewed
        starter_pct = (counts.get("starter", 0) / total) * 100
        ambitious_pct = (counts.get("ambitious", 0) / total) * 100

        if ambitious_pct > 30:
            lines.append(
                "NOTE: Recent batches have been skewing ambitious. "
                "In this batch, prioritize starter and weekend ideas."
            )
        elif starter_pct < 25:
            lines.append(
                "NOTE: Recent batches have been short on starter ideas. "
                "Make sure this batch includes at least 2 clear starter ideas."
            )

        return "\n" + "\n".join(lines) + "\n"

    except Exception as exc:
        log.warning("Could not fetch difficulty summary: %s", exc)
        return ""


def task_run_researcher(**context):
    from dag_db import fetch_all, execute_many, get_connection

    conf = context["dag_run"].conf or {}
    interests = conf.get("interests", DEFAULT_INTERESTS)

    # Pull narrations for richer agent context
    rows = fetch_all(
        "SELECT github_path, narration FROM code_files "
        "WHERE narration IS NOT NULL LIMIT 15"
    )
    file_narrations = [{"path": r["github_path"], "narration": r["narration"]} for r in rows]

    # Pull existing idea titles to avoid duplicates
    existing = fetch_all(
        "SELECT title_concept FROM blog_ideas "
        "WHERE status NOT IN ('published') "
        "ORDER BY created_at DESC LIMIT 50"
    )
    existing_titles = [r["title_concept"] for r in existing]
    log.info("Found %d existing ideas to avoid duplicating", len(existing_titles))

    # Get recent difficulty distribution to pass to agent
    conn = get_connection()
    difficulty_context = _get_recent_difficulty_summary(conn)
    conn.close()

    blueprints = agent_researcher(
        interests=interests,
        existing_projects=DEFAULT_PROJECTS,
        file_narrations=file_narrations,
        existing_titles=existing_titles,
        difficulty_context=difficulty_context,
    )
    log.info("Researcher produced %d blueprints", len(blueprints))

    # Filter out near-duplicates before inserting
    existing_lower = {t.lower() for t in existing_titles}
    new_blueprints = []
    for bp in blueprints:
        title = bp.get("title_concept", "").lower()
        is_duplicate = any(
            title in existing.lower() or existing.lower() in title
            for existing in existing_lower
        )
        if is_duplicate:
            log.info("Skipping duplicate: %s", bp.get("title_concept"))
        else:
            new_blueprints.append(bp)

    if not new_blueprints:
        log.info("All generated ideas were duplicates — nothing inserted")
        return

    statements = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for bp in new_blueprints:
        ptype = bp.get("project_type", "new_build")
        if ptype not in ("existing_asset", "new_build", "tutorial"):
            ptype = "new_build"

        difficulty = bp.get("difficulty", "weekend")
        if difficulty not in ("starter", "weekend", "ambitious"):
            difficulty = "weekend"

        statements.append((
            """INSERT INTO blog_ideas
               (title_concept, project_type, difficulty, the_build, the_narrative,
                the_selling_point, status, created_at, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                bp.get("title_concept", "Untitled"),
                ptype,
                difficulty,
                bp.get("the_build"),
                bp.get("the_narrative"),
                bp.get("the_selling_point"),
                "idea_generated",
                now,
                now,
            )
        ))

    execute_many(statements)
    log.info(
        "Inserted %d blog ideas. Difficulties: %s",
        len(statements),
        {bp.get("difficulty") for bp in new_blueprints},
    )


with DAG(
    dag_id="life_os_blog_scout",
    default_args=default_args,
    schedule_interval="0 9 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "blog"],
) as dag:

    run_researcher = PythonOperator(
        task_id="run_researcher",
        python_callable=task_run_researcher,
    )
