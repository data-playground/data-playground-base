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
import asyncio
import logging
from datetime import datetime, timedelta

sys.path.insert(0, '/opt/airflow/project')
sys.path.insert(0, '/opt/airflow/project/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator
from agents.blog_agents import agent_researcher

log = logging.getLogger(__name__)

default_args = {
    "owner": "life_os",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

# ── Update these to match your current projects ────────────────────────────────
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


def task_run_researcher(**context):
    import json
    from secrets import get_key
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from models import BlogIdea, BlogIdeaStatus, BlogProjectType

    # Allow conf overrides for manual triggers
    conf = context["dag_run"].conf or {}
    interests = conf.get("interests", DEFAULT_INTERESTS)

    # Pull file narrations from DB for richer context (optional but valuable)
    async def get_narrations():
        mdb_json = json.loads(get_key("MariaDB"))
        db_url = f"mysql+asyncmy://data_playground:{mdb_json['password']}@db:3306/jobs"
        engine = create_async_engine(db_url)
        session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        from sqlalchemy import select
        from models import CodeFile

        async with session_maker() as session:
            result = await session.execute(
                select(CodeFile.github_path, CodeFile.narration)
                .where(CodeFile.narration.isnot(None))
                .limit(15)  # Keep prompt size reasonable
            )
            return [{"path": r[0], "narration": r[1]} for r in result.all()]

    file_narrations = asyncio.run(get_narrations())

    # Run the Researcher agent
    blueprints = agent_researcher(
        interests=interests,
        existing_projects=DEFAULT_PROJECTS,
        file_narrations=file_narrations,
    )
    log.info("Researcher produced %d blueprints", len(blueprints))

    # Insert into blog_ideas
    async def insert_ideas():
        mdb_json = json.loads(get_key("MariaDB"))
        db_url = f"mysql+asyncmy://data_playground:{mdb_json['password']}@db:3306/jobs"
        engine = create_async_engine(db_url)
        session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with session_maker() as session:
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

            await session.commit()
            log.info("Inserted %d blog ideas into backlog", inserted)

    asyncio.run(insert_ideas())


with DAG(
    dag_id="life_os_blog_scout",
    default_args=default_args,
    schedule_interval="0 9 * * 1,4",  # Mon + Thu at 9am
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "blog"],
    doc_md=__doc__,
) as dag:

    run_researcher = PythonOperator(
        task_id="run_researcher",
        python_callable=task_run_researcher,
    )
