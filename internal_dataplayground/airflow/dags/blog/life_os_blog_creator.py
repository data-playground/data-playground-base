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
    from agents.blog_agents import agent_ghostwriter

    idea_id = context["dag_run"].conf.get("idea_id")
    idea = fetch_one("SELECT * FROM blog_ideas WHERE id = %s", (idea_id,))

    # Pull narration from linked code_file if exists
    code_narrative = ""
    if idea.get("code_file_id"):
        cf = fetch_one("SELECT narration, file_name FROM code_files WHERE id = %s",
                       (idea["code_file_id"],))
        if cf:
            code_narrative = cf.get("narration") or ""
    
    # Fall back to manually entered code_content from HITL evidence step
    if not code_narrative and idea.get("code_content"):
        code_narrative = idea["code_content"]

    draft = agent_ghostwriter(
        blueprint={
            "title_concept": idea["title_concept"],
            "the_build": idea.get("the_build") or "",
            "the_narrative": idea.get("the_narrative") or "",
            "the_selling_point": idea.get("the_selling_point") or "",
        },
        author_notes=idea.get("author_notes") or "",
        code_narrative=code_narrative,
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