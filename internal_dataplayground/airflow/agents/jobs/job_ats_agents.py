# airflow/agents/job_ats_agents.py
"""
ATS (Applicant Tracking System) job board fetchers — Greenhouse + Lever.

Unlike job_agents.py's LinkedIn scraper, these hit official public JSON
APIs — no HTML parsing, no anti-bot risk, no login required. The tradeoff
is you need to know each company's board token/slug in advance (there's no
public directory of which companies use which ATS) — that's what the
watched_companies table + the /jobs/config UI are for.

  Greenhouse: https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true
  Lever:      https://api.lever.co/v0/postings/{slug}?mode=json

Both are free and unauthenticated, and both return full job descriptions
inline — no separate "detail page" fetch needed the way the LinkedIn
scraper requires via get_job_details().

Output shape matches job_agents.search_linkedin_jobs() with two additions:
`source` and `external_ref`. Greenhouse job IDs are numeric; Lever's are
UUIDs — neither fits the legacy `linkedin_jobs.job_id` BigInteger column,
which is why external_ref exists as a separate VARCHAR column (see the
schema migration in this same batch of changes).
"""
import logging
import re

import requests

log = logging.getLogger(__name__)

_TIMEOUT = 15


def _strip_html(html: str) -> str:
    """
    Greenhouse/Lever descriptions come back as HTML. Strip tags for a plain
    text version consistent with what the LinkedIn scraper produces (and
    what the Gemini scoring prompt in job_agents.py expects).
    """
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", "\n", html)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def fetch_greenhouse_jobs(company_name: str, slug: str) -> list[dict]:
    """
    Fetches all open postings for one Greenhouse-hosted company.
    `slug` is the board token from the company's Greenhouse URL, e.g. for
    boards.greenhouse.io/anthropic the slug is "anthropic".
    """
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
    try:
        resp = requests.get(url, params={"content": "true"}, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("Greenhouse fetch failed for %s (slug=%s): %s", company_name, slug, exc)
        return []

    jobs = []
    for item in resp.json().get("jobs", []):
        jobs.append({
            "job_id":       None,   # legacy numeric column doesn't apply to ATS sources
            "source":       "greenhouse",
            "external_ref": str(item["id"]),
            "job_title":    item.get("title", "Unknown"),
            "company_name": company_name,
            "location":     (item.get("location") or {}).get("name"),
            "post_date":    (item.get("updated_at") or "")[:10] or None,
            "job_link":     item.get("absolute_url"),
            "description":  _strip_html(item.get("content", "")),
            "salary":       None,  # not exposed by Greenhouse's public board API
            "job_search":   f"ats:{company_name}",
        })
    return jobs


def fetch_lever_jobs(company_name: str, slug: str) -> list[dict]:
    """
    Fetches all open postings for one Lever-hosted company.
    `slug` is the token from the company's Lever URL, e.g. for
    jobs.lever.co/figma the slug is "figma".
    """
    url = f"https://api.lever.co/v0/postings/{slug}"
    try:
        resp = requests.get(url, params={"mode": "json"}, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("Lever fetch failed for %s (slug=%s): %s", company_name, slug, exc)
        return []

    jobs = []
    for item in resp.json():
        categories = item.get("categories", {}) or {}
        salary_range = item.get("salaryRange") or {}
        salary = None
        if salary_range.get("min") and salary_range.get("max"):
            salary = (
                f"{salary_range['min']}-{salary_range['max']} "
                f"{salary_range.get('currency', '')}"
            ).strip()

        jobs.append({
            "job_id":       None,
            "source":       "lever",
            "external_ref": item.get("id"),  # Lever IDs are UUID strings, not numeric
            "job_title":    item.get("text", "Unknown"),
            "company_name": company_name,
            "location":     categories.get("location"),
            "post_date":    None,  # Lever doesn't expose a reliable posted-date field
            "job_link":     item.get("hostedUrl"),
            "description":  _strip_html(item.get("descriptionPlain") or item.get("description", "")),
            "salary":       salary,
            "job_search":   f"ats:{company_name}",
        })
    return jobs


def fetch_all_watched_companies(companies: list[dict]) -> list[dict]:
    """
    Convenience wrapper for the DAG task: takes rows shaped like
    watched_companies (company_name, greenhouse_slug, lever_slug) and
    fetches from whichever board(s) each company has a slug for.
    """
    all_jobs = []
    for company in companies:
        name = company["company_name"]
        if company.get("greenhouse_slug"):
            found = fetch_greenhouse_jobs(name, company["greenhouse_slug"])
            log.info("Greenhouse: %s returned %d postings", name, len(found))
            all_jobs.extend(found)
        if company.get("lever_slug"):
            found = fetch_lever_jobs(name, company["lever_slug"])
            log.info("Lever: %s returned %d postings", name, len(found))
            all_jobs.extend(found)
    return all_jobs
