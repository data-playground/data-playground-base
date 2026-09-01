# airflow/dags/life_os_readme_writer.py
"""
README Writer DAG — triggered from LifeOS Code Intelligence UI.
Conf required: {"project_id": 1}
Conf optional: {"folder_path": "internal_dataplayground/routers"}
"""
import sys
import logging
from datetime import datetime

sys.path.insert(0, '/opt/airflow/project')
sys.path.insert(0, '/opt/airflow/project/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator
from agents.blog_agents import agent_readme_writer

log = logging.getLogger(__name__)

default_args = {"owner": "life_os", "retries": 0, "email_on_failure": False}


def task_write_readme(**context):
    from dag_db import fetch_one, fetch_all, execute

    conf = context["dag_run"].conf or {}
    project_id = conf.get("project_id")
    folder_path = conf.get("folder_path", "").strip().rstrip("/")

    if not project_id:
        raise ValueError("project_id required")

    project = fetch_one("SELECT * FROM code_projects WHERE id = %s", (project_id,))
    if not project:
        raise ValueError(f"Project {project_id} not found")

    # Fetch narrated files, optionally filtered by folder_path
    if folder_path:
        files = fetch_all(
            "SELECT file_name, github_path, narration FROM code_files "
            "WHERE project_id = %s AND narration IS NOT NULL "
            "AND (github_path LIKE %s OR github_path = %s)",
            (project_id, folder_path + "/%", folder_path)
        )
    else:
        files = fetch_all(
            "SELECT file_name, github_path, narration FROM code_files "
            "WHERE project_id = %s AND narration IS NOT NULL",
            (project_id,)
        )

    if not files:
        raise ValueError(
            f"No narrated files found{'under ' + folder_path if folder_path else ''}. "
            "Narrate files first."
        )

    log.info("Writing README for project %d from %d files (folder: %s)",
             project_id, len(files), folder_path or "all")

    project_name = project["project_name"]
    if folder_path:
        project_name = f"{project_name} / {folder_path.split('/')[-1]}"

    file_summaries = [
        {"path": f["github_path"], "narration": f["narration"]}
        for f in files
    ]

    readme = agent_readme_writer(
        project_name=project_name,
        file_summaries=file_summaries,
        description=project.get("description") or "",
    )

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    execute(
        """UPDATE code_projects
           SET readme_md = %s, readme_status = %s, readme_generated_at = %s, updated_at = %s
           WHERE id = %s""",
        (readme, "draft", now, now, project_id)
    )
    log.info("✓ README written for project %d", project_id)


with DAG(
    dag_id="life_os_readme_writer",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "code_intel"],
) as dag:
    PythonOperator(task_id="write_readme", python_callable=task_write_readme)