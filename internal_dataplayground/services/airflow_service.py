# services/airflow_service.py
"""
Shared Airflow DAG trigger helper.

Used by:
  - routers/ci_readme.py  (README Writer DAG)
  - routers/ci_files.py   (Narrate / Comment / Improve batch DAGs)
  - routers/blog.py       (Scout / Creator / Finalizer / Idea Expander DAGs)

All callers use _trigger_airflow(dag_id, conf) and receive back the run_id
string. Auth is Basic with the Airflow admin password pulled from GCP Secret
Manager on every call — no caching, so a rotated secret takes effect immediately.
"""

import base64
import logging
import httpx

from gcp_secrets import get_key

log = logging.getLogger(__name__)

AIRFLOW_BASE = "http://airflow-webserver:8080/api/v1"
AIRFLOW_USER = "admin"


def _airflow_headers() -> dict:
    """Build Basic-auth headers using the Airflow admin password from GCP."""
    password = get_key("Airflow-Admin-Password")
    token = base64.b64encode(f"{AIRFLOW_USER}:{password}".encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
    }


async def trigger_airflow(dag_id: str, conf: dict | None = None) -> str:
    """
    Trigger an Airflow DAG and return the run_id.

    Args:
        dag_id: The DAG identifier string (e.g. "life_os_blog_creator").
        conf:   Optional dict passed as DAG run configuration.
                Keys and values must be JSON-serialisable.

    Returns:
        dag_run_id string from the Airflow API response.

    Raises:
        httpx.HTTPStatusError: On non-2xx responses (excluding 409 which means
                               the DAG is already running — still returns run_id).
    """
    payload: dict = {"conf": conf or {}}

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{AIRFLOW_BASE}/dags/{dag_id}/dagRuns",
            headers=_airflow_headers(),
            json=payload,
        )

        # 409 = DAG already running — Airflow still returns the run data
        if resp.status_code not in (200, 201, 409):
            log.error(
                "Airflow trigger failed for %s: HTTP %s — %s",
                dag_id, resp.status_code, resp.text,
            )
            resp.raise_for_status()

        data = resp.json()
        run_id: str = data.get("dag_run_id", "unknown")
        log.info("Triggered DAG %s → run_id: %s", dag_id, run_id)
        return run_id
