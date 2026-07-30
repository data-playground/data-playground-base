# airflow/agents/job_scout_health.py
"""
"Is the scrape actually working" health tracking for both Job Scout DAGs
(life_os_job_scout.py for LinkedIn, life_os_job_scout_ats.py for
Greenhouse/Lever).

Every run of either DAG writes one row to job_scout_run_log via log_run().
life_os_daily_digest.py reads the latest state via get_health_summary() to
decide whether to include a warning line in the email — one source of
truth for both the digest and the /jobs/config Settings page (which reads
the same table independently via SQLAlchemy, since the FastAPI container
doesn't have dag_db/pymysql available — see routers/job_config.py).

CONSECUTIVE_ZERO_THRESHOLD: how many back-to-back runs with items_found == 0
before this is treated as a real signal rather than a coincidence. A single
empty run happens sometimes (LinkedIn hiccup, or a watched company
genuinely has zero open reqs that hour) — three in a row is not a
coincidence.
"""
import logging
from datetime import datetime

log = logging.getLogger(__name__)

CONSECUTIVE_ZERO_THRESHOLD = 3


def log_run(
    dag_id: str,
    items_attempted: int,
    items_found: int,
    new_items: int,
    items_loaded: int,
) -> None:
    """Writes one row to job_scout_run_log and flags a warning if this run
    extends a streak of zero-postings-found runs for this DAG."""
    from dag_db import fetch_all, execute

    status = "ok"
    message = None

    if items_found == 0:
        recent = fetch_all(
            "SELECT items_found FROM job_scout_run_log "
            "WHERE dag_id = %s ORDER BY run_at DESC LIMIT %s",
            (dag_id, CONSECUTIVE_ZERO_THRESHOLD - 1),
        )
        prior_zero_streak = len(recent) == CONSECUTIVE_ZERO_THRESHOLD - 1 and all(
            r["items_found"] == 0 for r in recent
        )
        if prior_zero_streak:
            status = "warning"
            message = (
                f"{CONSECUTIVE_ZERO_THRESHOLD} consecutive runs with 0 postings found. "
                "For the LinkedIn DAG this usually means the scraper is being blocked "
                "or the page markup changed. For the ATS DAG it can also just mean "
                "every watched company currently has zero open roles."
            )
            log.warning("%s: %s", dag_id, message)

    execute(
        "INSERT INTO job_scout_run_log "
        "(dag_id, run_at, items_attempted, items_found, new_items, items_loaded, status, message) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        (dag_id, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
         items_attempted, items_found, new_items, items_loaded, status, message),
    )


def get_health_summary() -> list[dict]:
    """Most recent run + status for each Job Scout DAG. Used by the digest."""
    from dag_db import fetch_all

    return fetch_all(
        """
        SELECT t1.dag_id, t1.run_at, t1.items_found, t1.new_items,
               t1.items_loaded, t1.status, t1.message
        FROM job_scout_run_log t1
        INNER JOIN (
            SELECT dag_id, MAX(run_at) AS max_run_at
            FROM job_scout_run_log
            GROUP BY dag_id
        ) t2 ON t1.dag_id = t2.dag_id AND t1.run_at = t2.max_run_at
        """
    )
