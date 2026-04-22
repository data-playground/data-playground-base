# airflow/dags/life_os_code_narrate.py
"""
Code Narrator DAG — triggered from LifeOS Code Intelligence UI.
Accepts a list of file IDs and narrates each one sequentially.
Conf required: {"file_ids": [1, 2, 3], "project_id": 42}
"""
import sys
import logging
from datetime import datetime, timedelta

sys.path.insert(0, '/opt/airflow/project')
sys.path.insert(0, '/opt/airflow/project/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator
from agents.blog_agents import agent_code_narrator

log = logging.getLogger(__name__)

default_args = {
    "owner": "life_os",
    "retries": 0,
    "email_on_failure": False,
}


def task_narrate_files(**context):
    from dag_db import fetch_one, execute, fetch_all

    conf = context["dag_run"].conf or {}
    file_ids = conf.get("file_ids", [])
    if not file_ids:
        raise ValueError("file_ids list is required in conf")

    log.info("Narrating %d files: %s", len(file_ids), file_ids)

    for file_id in file_ids:
        file = fetch_one("SELECT * FROM code_files WHERE id = %s", (file_id,))
        if not file:
            log.warning("File %d not found, skipping", file_id)
            continue

        if not file.get("raw_code"):
            log.warning("File %d has no raw_code — pull it first", file_id)
            continue

        # Get project description for context
        project = fetch_one("SELECT * FROM code_projects WHERE id = %s", (file["project_id"],))
        readme_context = (project.get("readme_md") or project.get("description") or "") if project else ""

        log.info("Narrating %s", file["file_name"])
        try:
            narration = agent_code_narrator(
                code_content=file["raw_code"],
                file_name=file["file_name"],
                readme_context=readme_context,
            )
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            execute(
                "UPDATE code_files SET narration = %s, narration_generated_at = %s, updated_at = %s WHERE id = %s",
                (narration, now, now, file_id)
            )
            log.info("✓ Narrated %s", file["file_name"])
        except Exception as exc:
            log.error("Failed to narrate file %d (%s): %s", file_id, file["file_name"], exc)
            # Continue with next file rather than failing the whole DAG


with DAG(
    dag_id="life_os_code_narrate",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "code_intel"],
) as dag:
    PythonOperator(task_id="narrate_files", python_callable=task_narrate_files)