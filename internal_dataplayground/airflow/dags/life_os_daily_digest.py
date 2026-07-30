# airflow/dags/life_os_daily_digest.py
"""
Daily Digest DAG — the first piece of what's meant to grow into a broader
"newsletter of things that matter" across Life OS modules. For now it
covers Job Scout only: new high-fit jobs found in the last day, plus scrape
health from both Job Scout DAGs (see job_scout_health.py) — so a quiet or
blocked scraper surfaces here instead of silently doing nothing until you
happen to check the Settings page.

Runs once daily. Sends via Gmail SMTP (agents/email_client.py) to yourself.
Skips sending entirely if there's nothing to report (no new high-fit jobs
AND no health warnings) — no point in a daily "nothing happened" email.

Conf (optional):
  {"min_fit_score": 85, "lookback_days": 1}
"""
import sys
import logging
from datetime import datetime, timedelta

sys.path.insert(0, '/opt/airflow/project')
sys.path.insert(0, '/opt/airflow/project/airflow')

from airflow import DAG
from airflow.operators.python import PythonOperator

log = logging.getLogger(__name__)

DEFAULT_MIN_FIT_SCORE = 85
DEFAULT_LOOKBACK_DAYS = 1

default_args = {
    "owner": "life_os",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


def _render_digest_html(jobs: list[dict], health: list[dict]) -> str:
    """Self-contained, inline-styled HTML — email clients ignore external/linked CSS."""
    job_rows = "".join(
        f"""
        <tr>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e5ea;">
            <div style="font-weight:600;font-size:14px;color:#18182e;">{j['job_title']}</div>
            <div style="font-size:12px;color:#70708a;margin-top:2px;">{j['company_name'] or '—'} · {j['location'] or '—'}</div>
          </td>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e5ea;text-align:center;">
            <span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:700;
                         background:{'#e6f9f0' if j['fit_score'] >= 90 else '#fff4e0'};
                         color:{'#00a870' if j['fit_score'] >= 90 else '#b8790a'};">
              {j['fit_score']}
            </span>
          </td>
          <td style="padding:10px 12px;border-bottom:1px solid #e5e5ea;text-align:right;">
            <a href="{j['job_link']}" style="font-size:12px;color:#5548e8;text-decoration:none;">View →</a>
          </td>
        </tr>
        """
        for j in jobs
    )

    health_rows = "".join(
        f"""
        <div style="padding:10px 14px;margin-bottom:8px;border-radius:6px;
                    background:{'#fff4e0' if h['status'] == 'warning' else '#f5f4fb'};
                    border-left:3px solid {'#e8a020' if h['status'] == 'warning' else '#7c6fff'};">
          <div style="font-size:12px;font-weight:600;color:#18182e;">{h['dag_id']}</div>
          <div style="font-size:11px;color:#54527a;margin-top:2px;">
            last run {h['run_at']} — {h['items_found']} found, {h['new_items']} new, {h['items_loaded']} loaded
          </div>
          {f'<div style="font-size:11px;color:#b8790a;margin-top:4px;">⚠ {h["message"]}</div>' if h.get('message') else ''}
        </div>
        """
        for h in health
    )

    jobs_section = (
        f'<table style="width:100%;border-collapse:collapse;">{job_rows}</table>'
        if jobs else
        '<p style="font-size:13px;color:#70708a;">No new jobs above the fit threshold today.</p>'
    )

    return f"""
    <div style="font-family:-apple-system,Segoe UI,sans-serif;max-width:600px;margin:0 auto;">
      <h2 style="font-size:18px;color:#18182e;">Life OS — Job Scout Digest</h2>

      <h3 style="font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#70708a;margin-top:24px;">
        New high-fit jobs
      </h3>
      {jobs_section}

      <h3 style="font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#70708a;margin-top:24px;">
        Scrape health
      </h3>
      {health_rows or '<p style="font-size:13px;color:#70708a;">No run history yet.</p>'}
    </div>
    """


def task_build_and_send(**context):
    from dag_db import fetch_all
    from agents.job_scout_health import get_health_summary
    from agents.email_client import send_email

    conf = context["dag_run"].conf or {}
    min_fit = conf.get("min_fit_score", DEFAULT_MIN_FIT_SCORE)
    lookback_days = conf.get("lookback_days", DEFAULT_LOOKBACK_DAYS)

    cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    jobs = fetch_all(
        "SELECT job_title, company_name, location, fit_score, job_link "
        "FROM linkedin_jobs WHERE search_date >= %s AND fit_score >= %s "
        "ORDER BY fit_score DESC LIMIT 25",
        (cutoff, min_fit),
    )
    health = get_health_summary()

    if not jobs and not any(h["status"] == "warning" for h in health):
        log.info("Nothing to report today — skipping email")
        return

    html = _render_digest_html(jobs, health)
    subject = f"Job Scout Digest — {len(jobs)} new high-fit job(s)"
    send_email(subject, html)
    log.info("Digest sent: %d jobs, %d health rows", len(jobs), len(health))


with DAG(
    dag_id="life_os_daily_digest",
    default_args=default_args,
    schedule_interval="0 13 * * *",  # 1pm UTC ≈ 8-9am US Eastern
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["life_os", "jobs", "digest"],
    doc_md="""
## Daily Digest DAG

New high-fit jobs from the last day + scrape health for both Job Scout
DAGs, emailed via Gmail SMTP. Skips sending if there's nothing to report.

**Manual trigger with different thresholds:**
```json
{"min_fit_score": 80, "lookback_days": 2}
```
    """,
) as dag:
    PythonOperator(task_id="build_and_send", python_callable=task_build_and_send)
