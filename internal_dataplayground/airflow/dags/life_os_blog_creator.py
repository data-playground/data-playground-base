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
    from dag_db import fetch_one, execute

    idea_id = context["dag_run"].conf.get("idea_id")
    if not idea_id:
        raise ValueError("idea_id is required in DAG conf")

    idea = fetch_one("SELECT * FROM blog_ideas WHERE id = %s", (idea_id,))
    if not idea:
        raise ValueError(f"BlogIdea {idea_id} not found")

    # Pull narration from linked code file if present
    narration = ""
    if idea.get("code_file_id"):
        code_file = fetch_one(
            "SELECT narration FROM code_files WHERE id = %s",
            (idea["code_file_id"],)
        )
        if code_file and code_file.get("narration"):
            narration = code_file["narration"]

    blueprint = {
        "title_concept": idea["title_concept"],
        "the_build": idea.get("the_build") or "",
        "the_narrative": idea.get("the_narrative") or "",
        "the_selling_point": idea.get("the_selling_point") or "",
    }

    log.info("Running Ghostwriter for idea %d: %s", idea_id, idea["title_concept"])
    draft = agent_ghostwriter(
        blueprint=blueprint,
        author_notes=idea.get("author_notes") or "",
        code_narrative=narration,
    )

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    execute(
        "UPDATE blog_ideas SET draft_v1 = %s, status = %s, updated_at = %s WHERE id = %s",
        (draft, "waiting_for_review", now, idea_id)
    )
    log.info("draft_v1 written for idea %d, status → waiting_for_review", idea_id)


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