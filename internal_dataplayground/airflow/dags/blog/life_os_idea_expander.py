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
    "retries": 3,
    "retry_delay": timedelta(minutes=2),  # gives Gemini time to recover
    "email_on_failure": False,
}

def task_expand_idea(**context):
    from dag_db import fetch_one, fetch_all, execute
    from agents.blog_agents import agent_idea_expander

    idea_id = context["dag_run"].conf.get("idea_id")
    idea = fetch_one("SELECT * FROM blog_ideas WHERE id = %s", (idea_id,))
    raw = idea.get("raw_idea_input") or idea.get("title_concept", "")

    # If idea has a linked code_file, pull its narration and enrich the input
    narration_context = ""
    if idea.get("code_file_id"):
        cf = fetch_one("SELECT narration, file_name FROM code_files WHERE id = %s",
                       (idea["code_file_id"],))
        if cf and cf.get("narration"):
            narration_context = f"\n\nRelated code file ({cf['file_name']}):\n{cf['narration']}"
    elif idea.get("code_project_id"):
        # Pull all narrations for the project
        files = fetch_all(
            "SELECT file_name, narration FROM code_files "
            "WHERE project_id = %s AND narration IS NOT NULL LIMIT 5",
            (idea["code_project_id"],)
        )
        if files:
            narration_context = "\n\nRelated project files:\n" + "\n\n".join(
                f"{f['file_name']}: {f['narration'][:400]}" for f in files
            )

    blueprint = agent_idea_expander(raw + narration_context)

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    execute(
        """UPDATE blog_ideas
           SET title_concept = %s,
               project_type  = %s,
               the_build     = %s,
               the_narrative = %s,
               the_selling_point = %s,
               updated_at    = %s
           WHERE id = %s""",
        (
            blueprint.get("title_concept", idea["title_concept"]),
            blueprint.get("project_type", "new_build"),
            blueprint.get("the_build"),
            blueprint.get("the_narrative"),
            blueprint.get("the_selling_point"),
            now,
            idea_id,
        )
    )
    log.info("Idea %d enriched: %s", idea_id, blueprint.get("title_concept"))

with DAG(
    dag_id="life_os_idea_expander",
    default_args=default_args,
    schedule_interval=None,  # manual/triggered only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "blog"],
) as dag:
    expand = PythonOperator(
        task_id="expand_idea",
        python_callable=task_expand_idea,
    )