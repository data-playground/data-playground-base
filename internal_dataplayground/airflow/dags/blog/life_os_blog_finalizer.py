# airflow/dags/life_os_blog_finalizer.py
# ARCHITECTURAL RULE: This DAG file must never import from models.py,
# database.py, or any FastAPI router. All database access uses dag_db.py.
"""
Blog Finalizer DAG  (Trigger 2 of 2)
──────────────────────────────────────
Triggered from the LifeOS Blog UI after you complete your review
of draft_v1 and click "Finalize".

Runs: Refiner → Editor (sequential)
Sets status: ready_to_publish

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

log = logging.getLogger(__name__)

default_args = {
    "owner": "life_os",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def _parse_editor_output(raw: str) -> tuple[str, str, str, str]:
    try:
        parts = raw.split("---")
        if len(parts) < 3:
            return "", "", "", raw.strip()
        meta_block = parts[1].strip()
        article = "---".join(parts[2:]).strip()
        title, desc, tags = "", "", ""
        for line in meta_block.splitlines():
            line = line.strip()
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip()
            elif line.lower().startswith("meta description:"):
                desc = line.split(":", 1)[1].strip()
            elif line.lower().startswith("tags:"):
                tags = line.split(":", 1)[1].strip()
        return title, desc, tags, article
    except Exception as exc:
        log.warning("Editor output parsing failed: %s", exc)
        return "", "", "", raw


def task_refiner(**context):
    # ARCHITECTURAL RULE: raw SQL only — no ORM imports
    from dag_db import fetch_one, execute
    from agents.blog_agents import agent_refiner

    idea_id = context["dag_run"].conf.get("idea_id")
    idea = fetch_one("SELECT * FROM blog_ideas WHERE id = %s", (idea_id,))
    if not idea:
        raise ValueError(f"BlogIdea {idea_id} not found")
    if not idea.get("draft_v1"):
        raise ValueError(f"BlogIdea {idea_id} has no draft_v1")

    log.info("Running Refiner for idea %d", idea_id)
    refined = agent_refiner(
        original_draft=idea["draft_v1"],
        user_feedback=idea.get("user_review_notes") or "No specific feedback provided.",
    )

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    execute(
        "UPDATE blog_ideas SET draft_v2 = %s, status = %s, updated_at = %s WHERE id = %s",
        (refined, "review_completed", now, idea_id),
    )
    log.info("draft_v2 written for idea %d", idea_id)


def task_editor(**context):
    # ARCHITECTURAL RULE: raw SQL only — no ORM imports
    from dag_db import fetch_one, execute
    from agents.blog_agents import agent_editor

    idea_id = context["dag_run"].conf.get("idea_id")
    idea = fetch_one("SELECT * FROM blog_ideas WHERE id = %s", (idea_id,))
    if not idea:
        raise ValueError(f"BlogIdea {idea_id} not found")

    source = idea.get("draft_v2") or idea.get("draft_v1") or ""
    if not source:
        raise ValueError(f"BlogIdea {idea_id} has no draft to edit")

    log.info("Running Editor for idea %d", idea_id)
    final_output = agent_editor(draft_content=source)
    seo_title, seo_desc, seo_tags, article = _parse_editor_output(final_output)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    execute(
        """UPDATE blog_ideas
           SET final_article    = %s,
               seo_title        = %s,
               seo_description  = %s,
               seo_tags         = %s,
               status           = %s,
               updated_at       = %s
           WHERE id = %s""",
        (
            article,
            seo_title or idea.get("title_concept", ""),
            seo_desc,
            seo_tags,
            "ready_to_publish",
            now,
            idea_id,
        ),
    )
    log.info("Final article written for idea %d", idea_id)


with DAG(
    dag_id="life_os_blog_finalizer",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "blog"],
) as dag:

    refine = PythonOperator(task_id="refiner", python_callable=task_refiner)
    edit   = PythonOperator(task_id="editor",  python_callable=task_editor)
    refine >> edit
