# airflow/dags/life_os_code_improve.py
"""
Code Improver DAG — triggered from LifeOS Code Intelligence UI.
Conf required: {"file_ids": [1, 2, 3]}
"""
import sys
import logging
from datetime import datetime

sys.path.insert(0, '/opt/airflow/project')
sys.path.insert(0, '/opt/airflow/project/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator
from agents.blog_agents import agent_code_improver

log = logging.getLogger(__name__)

default_args = {"owner": "life_os", "retries": 0, "email_on_failure": False}


def task_improve_files(**context):
    from dag_db import fetch_one, execute

    file_ids = (context["dag_run"].conf or {}).get("file_ids", [])
    if not file_ids:
        raise ValueError("file_ids required")

    for file_id in file_ids:
        file = fetch_one("SELECT * FROM code_files WHERE id = %s", (file_id,))
        if not file or not file.get("raw_code"):
            log.warning("File %d not found or no raw_code, skipping", file_id)
            continue

        log.info("Improving %s", file["file_name"])
        try:
            notes = agent_code_improver(
                code_content=file["raw_code"],
                file_name=file["file_name"],
                narration=file.get("narration") or "",
            )
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            execute(
                """UPDATE code_files
                   SET improvement_notes = %s, improvement_generated_at = %s,
                       improvement_status = %s, updated_at = %s
                   WHERE id = %s""",
                (notes, now, "generated", now, file_id)
            )
            log.info("✓ Improved %s", file["file_name"])
        except Exception as exc:
            log.error("Failed to improve file %d: %s", file_id, exc)


with DAG(
    dag_id="life_os_code_improve",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "code_intel"],
) as dag:
    PythonOperator(task_id="improve_files", python_callable=task_improve_files)