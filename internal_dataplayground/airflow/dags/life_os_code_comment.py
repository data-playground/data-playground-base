# airflow/dags/life_os_code_comment.py
"""
Code Commenter DAG — triggered from LifeOS Code Intelligence UI.
Conf required: {"file_ids": [1, 2, 3]}
"""
import sys
import logging
from datetime import datetime, timedelta

sys.path.insert(0, '/opt/airflow/project')
sys.path.insert(0, '/opt/airflow/project/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator
from agents.blog_agents import agent_code_commenter

log = logging.getLogger(__name__)

default_args = {"owner": "life_os", "retries": 0, "email_on_failure": False}


def task_comment_files(**context):
    from dag_db import fetch_one, execute

    file_ids = (context["dag_run"].conf or {}).get("file_ids", [])
    if not file_ids:
        raise ValueError("file_ids required")

    for file_id in file_ids:
        file = fetch_one("SELECT * FROM code_files WHERE id = %s", (file_id,))
        if not file or not file.get("raw_code"):
            log.warning("File %d not found or no raw_code, skipping", file_id)
            continue

        log.info("Commenting %s", file["file_name"])
        try:
            commented = agent_code_commenter(
                code_content=file["raw_code"],
                file_name=file["file_name"],
            )
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            execute(
                """UPDATE code_files
                   SET commented_code = %s, commented_generated_at = %s,
                       commented_status = %s, updated_at = %s
                   WHERE id = %s""",
                (commented, now, "generated", now, file_id)
            )
            log.info("✓ Commented %s", file["file_name"])
        except Exception as exc:
            log.error("Failed to comment file %d: %s", file_id, exc)


with DAG(
    dag_id="life_os_code_comment",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "code_intel"],
) as dag:
    PythonOperator(task_id="comment_files", python_callable=task_comment_files)