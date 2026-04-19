# airflow/dags/life_os_blog_creator.py
"""
Blog Creator DAG  (Trigger 1 of 2)
────────────────────────────────────
Triggered manually from the LifeOS Blog UI when you click
"Trigger Airflow Creator" on an idea card.

Runs: Ghostwriter only
Sets status: WAITING_FOR_REVIEW

After this DAG completes:
  1. You review draft_v1 in the LifeOS UI
  2. You add review notes
  3. You click "Finalize" which triggers life_os_blog_finalizer

Conf (required):
  {"idea_id": 42}
"""

import sys
import logging
from datetime import datetime, timedelta

sys.path.insert(0, '/opt/airflow/project')
sys.path.insert(0, '/opt/airflow/project/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator
from agents.blog_agents import agent_ghostwriter

log = logging.getLogger(__name__)

default_args = {
    "owner": "life_os",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def task_ghostwriter(**context):
    from dag_db import get_sync_session
    from models import BlogIdea, BlogIdeaStatus, CodeFile

    idea_id = context["dag_run"].conf.get("idea_id")
    if not idea_id:
        raise ValueError("idea_id is required in DAG conf")

    session = get_sync_session()
    try:
        idea = session.get(BlogIdea, idea_id)
        if not idea:
            raise ValueError(f"BlogIdea {idea_id} not found")

        narration = ""
        if idea.code_file_id:
            code_file = session.get(CodeFile, idea.code_file_id)
            if code_file and code_file.narration:
                narration = code_file.narration

        blueprint = {
            "title_concept": idea.title_concept,
            "the_build": idea.the_build or "",
            "the_narrative": idea.the_narrative or "",
            "the_selling_point": idea.the_selling_point or "",
        }

        log.info("Running Ghostwriter for idea %d: %s", idea_id, idea.title_concept)
        draft = agent_ghostwriter(
            blueprint=blueprint,
            author_notes=idea.author_notes or "",
            code_narrative=narration,
        )

        idea.draft_v1 = draft
        idea.status = BlogIdeaStatus.WAITING_FOR_REVIEW
        session.commit()
        log.info("draft_v1 written for idea %d", idea_id)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


with DAG(
    dag_id="life_os_blog_creator",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "blog"],
) as dag:

    ghostwrite = PythonOperator(
        task_id="ghostwriter",
        python_callable=task_ghostwriter,
    )