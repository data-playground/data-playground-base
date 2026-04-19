# airflow/dags/life_os_blog_scout.py
"""
Blog Scout DAG
─────────────
Runs on a schedule (Mon + Thu 9am) and calls the Researcher agent
to generate 3 new blog blueprints, inserting them into blog_ideas.

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
    "FastAPI, async Python, MariaDB, Docker, GCP, Apache Airflow, "
    "data pipelines, multi-agent AI, personal productivity systems"
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


def task_run_researcher(**context):
    from dag_db import get_sync_session
    from models import BlogIdea, BlogIdeaStatus, BlogProjectType, CodeFile

    conf = context["dag_run"].conf or {}
    interests = conf.get("interests", DEFAULT_INTERESTS)

    # Pull narrations from DB using sync session
    session = get_sync_session()
    try:
        rows = session.query(
            CodeFile.github_path, CodeFile.narration
        ).filter(
            CodeFile.narration.isnot(None)
        ).limit(15).all()
        file_narrations = [{"path": r[0], "narration": r[1]} for r in rows]
    finally:
        session.close()

    blueprints = agent_researcher(
        interests=interests,
        existing_projects=DEFAULT_PROJECTS,
        file_narrations=file_narrations,
    )
    log.info("Researcher produced %d blueprints", len(blueprints))

    session = get_sync_session()
    try:
        inserted = 0
        for bp in blueprints:
            try:
                ptype = BlogProjectType(bp.get("project_type", "new_build"))
            except ValueError:
                ptype = BlogProjectType.NEW_BUILD

            idea = BlogIdea(
                title_concept=bp.get("title_concept", "Untitled"),
                project_type=ptype,
                the_build=bp.get("the_build"),
                the_narrative=bp.get("the_narrative"),
                the_selling_point=bp.get("the_selling_point"),
                status=BlogIdeaStatus.IDEA_GENERATED,
            )
            session.add(idea)
            inserted += 1

        session.commit()
        log.info("Inserted %d blog ideas", inserted)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


with DAG(
    dag_id="life_os_blog_scout",
    default_args=default_args,
    schedule_interval="0 9 * * 1,4",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "blog"],
) as dag:

    run_researcher = PythonOperator(
        task_id="run_researcher",
        python_callable=task_run_researcher,
    )