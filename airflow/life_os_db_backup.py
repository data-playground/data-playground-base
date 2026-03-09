# dags/life_os_db_backup.py
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import subprocess
import gzip
import os
import logging
from pathlib import Path
from google.cloud import storage
from google.oauth2 import service_account

# =============================================================================
# CONFIGURATION
# All paths and settings live here. Update these if the project moves.
# =============================================================================
PROJECT_DIR = Path("/home/main-server/Github/data-playground-base/internal_dataplayground")
BACKUP_DIR = PROJECT_DIR / "db_backups"
GCP_BUCKET = "life-os-db-backups"
GCP_KEY_PATH = PROJECT_DIR / "impactful-post-292301-17bfe2bceb2c.json"
BASH_SCRIPT_PATH = PROJECT_DIR / "backup_db.sh"
DB_CONTAINER = "life_os_db"
DB_NAME = "jobs"
DB_USER = "root"
RETAIN_DAYS = 7

log = logging.getLogger(__name__)

# =============================================================================
# SHARED HELPERS
# Used by both the native Python path and the fallback bash path.
# =============================================================================

def get_db_password() -> str:
    """Read DB_ROOT_PASSWORD from the .env file."""
    env_path = PROJECT_DIR / ".env"
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("DB_ROOT_PASSWORD="):
                return line.split("=", 1)[1].strip()
    raise ValueError("DB_ROOT_PASSWORD not found in .env file")


# =============================================================================
# STRATEGY 1 — NATIVE PYTHON
# Preferred path. Uses subprocess + google-cloud-storage Python client.
# More granular retry logic and better Airflow observability.
# =============================================================================

def dump_database(**context) -> str:
    """
    Run mysqldump inside the Docker container and compress the output.
    Pushes the output filepath to XCom so the upload task can consume it.
    Falls back to bash strategy if this raises any exception.
    """
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"life_os_backup_{timestamp}.sql.gz"
    filepath = BACKUP_DIR / filename

    db_password = get_db_password()

    log.info(f"[Native] Starting mysqldump for database: {DB_NAME}")

    dump_cmd = [
        "docker", "exec", DB_CONTAINER,
        "mysqldump",
        f"-u{DB_USER}",
        f"-p{db_password}",
        "--single-transaction",
        "--routines",
        "--triggers",
        DB_NAME,
    ]

    with gzip.open(filepath, "wb") as gz_file:
        result = subprocess.run(
            dump_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            raise RuntimeError(f"mysqldump failed: {result.stderr.decode()}")

        gz_file.write(result.stdout)

    file_size = filepath.stat().st_size
    if file_size < 1000:
        raise RuntimeError(
            f"Dump file suspiciously small ({file_size} bytes). "
            "Possible empty dump or Docker connection failure."
        )

    log.info(f"[Native] Dump successful: {filename} ({file_size / 1024 / 1024:.2f} MB)")

    context["ti"].xcom_push(key="backup_filepath", value=str(filepath))
    context["ti"].xcom_push(key="backup_filename", value=filename)

    return str(filepath)


def upload_to_gcs(**context) -> None:
    """
    Upload the compressed dump to GCS using the service account key.
    Reads filepath from XCom set by dump_database.
    """
    filepath = context["ti"].xcom_pull(key="backup_filepath", task_ids="native_dump")
    filename = context["ti"].xcom_pull(key="backup_filename", task_ids="native_dump")

    log.info(f"[Native] Uploading {filename} to gs://{GCP_BUCKET}/...")

    credentials = service_account.Credentials.from_service_account_file(
        str(GCP_KEY_PATH)
    )
    client = storage.Client(
        project="impactful-post-292301",
        credentials=credentials,
    )

    bucket = client.bucket(GCP_BUCKET)
    blob = bucket.blob(filename)
    blob.upload_from_filename(filepath)

    log.info(f"[Native] Upload successful: gs://{GCP_BUCKET}/{filename}")


def cleanup_local_backups(**context) -> None:
    """
    Delete local backup files older than RETAIN_DAYS.
    Runs regardless of which strategy succeeded (native or bash).
    """
    log.info(f"[Cleanup] Removing local backups older than {RETAIN_DAYS} days...")

    cutoff = datetime.now().timestamp() - (RETAIN_DAYS * 86400)
    deleted = 0

    for f in BACKUP_DIR.glob("life_os_backup_*.sql.gz"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            log.info(f"[Cleanup] Deleted: {f.name}")
            deleted += 1

    log.info(f"[Cleanup] Done. {deleted} file(s) removed.")


# =============================================================================
# STRATEGY 2 — BASH FALLBACK
# Triggered automatically if the native Python dump or upload fails.
# Runs the existing backup_db.sh script as a safety net.
# =============================================================================

def check_native_success(**context) -> str:
    """
    BranchPythonOperator logic.
    Checks if both native tasks succeeded. If yes, skip bash fallback.
    If either failed, route to the bash fallback task.
    """
    ti = context["ti"]

    dump_state = ti.xcom_pull(task_ids="native_dump")
    upload_state = ti.xcom_pull(task_ids="native_upload")

    # If XCom values exist, native path completed successfully
    if dump_state and upload_state:
        log.info("[Branch] Native strategy succeeded. Skipping bash fallback.")
        return "skip_fallback"

    log.warning("[Branch] Native strategy incomplete. Routing to bash fallback.")
    return "bash_fallback"


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    "owner": "life_os",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,  # Flip to True once you configure SMTP in Airflow
}

with DAG(
    dag_id="life_os_db_backup",
    description=(
        "Nightly MariaDB backup. Tries native Python (subprocess + GCS client) first. "
        "Falls back to backup_db.sh automatically if native path fails."
    ),
    default_args=default_args,
    start_date=datetime(2026, 3, 8),
    schedule="0 2 * * *",  # Every night at 2:00 AM
    catchup=False,
    tags=["life_os", "backup", "infrastructure"],
) as dag:

    # ------------------------------------------------------------------
    # NATIVE PYTHON PATH (preferred)
    # ------------------------------------------------------------------
    native_dump = PythonOperator(
        task_id="native_dump",
        python_callable=dump_database,
        # If this fails, Airflow retries before moving on
    )

    native_upload = PythonOperator(
        task_id="native_upload",
        python_callable=upload_to_gcs,
    )

    # ------------------------------------------------------------------
    # BRANCH — Did native path succeed? Route accordingly.
    # ------------------------------------------------------------------
    branch = BranchPythonOperator(
        task_id="check_native_success",
        python_callable=check_native_success,
        trigger_rule="all_done",  # Run this even if upstream tasks failed
    )

    # ------------------------------------------------------------------
    # BASH FALLBACK PATH
    # Only runs if native_dump or native_upload failed.
    # ------------------------------------------------------------------
    bash_fallback = BashOperator(
        task_id="bash_fallback",
        bash_command=f"bash {BASH_SCRIPT_PATH}",
        trigger_rule="all_done",
    )

    # ------------------------------------------------------------------
    # SKIP NODE — Joins both paths back into cleanup
    # ------------------------------------------------------------------
    skip_fallback = EmptyOperator(
        task_id="skip_fallback",
    )

    # ------------------------------------------------------------------
    # CLEANUP — Runs regardless of which path was taken
    # ------------------------------------------------------------------
    cleanup = PythonOperator(
        task_id="cleanup_local_backups",
        python_callable=cleanup_local_backups,
        trigger_rule="none_failed_min_one_success",  # Run if at least one path worked
    )

    # ------------------------------------------------------------------
    # PIPELINE WIRING
    #
    #   native_dump → native_upload → branch ──► skip_fallback ──► cleanup
    #                                        └──► bash_fallback ──►
    # ------------------------------------------------------------------
    native_dump >> native_upload >> branch
    branch >> [skip_fallback, bash_fallback]
    [skip_fallback, bash_fallback] >> cleanup
    
"""

---

## How the Flow Works
native_dump → native_upload → check_native_success
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
              skip_fallback                   bash_fallback
              (native worked)                 (native failed)
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                            cleanup_local_backups
"""