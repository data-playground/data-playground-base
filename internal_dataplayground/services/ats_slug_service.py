# services/ats_slug_service.py
"""
Best-effort Greenhouse/Lever slug guesser for the Jobs > Config "Add Company"
form. There's no public directory mapping companies to their ATS board —
this just probes a couple of plausible slug variants and reports what
returns a valid (200) response.

Lives under services/ (FastAPI-only), not airflow/agents/, because it's
only ever called from routers/job_config.py — the DAGs never need it, and
this keeps the DAG/FastAPI import boundary clean per CONTRIBUTING.md.
"""
import logging
import re

import httpx

log = logging.getLogger(__name__)

_TIMEOUT = 8.0
_SUFFIX_PATTERN = re.compile(r"\b(inc|llc|corp|corporation|co|ltd)\.?\b")


def _slug_candidates(company_name: str) -> list[str]:
    """Generates a short list of plausible board slugs from a company name."""
    base = _SUFFIX_PATTERN.sub("", company_name.lower()).strip()
    no_space = re.sub(r"[^a-z0-9]+", "", base)
    hyphenated = re.sub(r"[^a-z0-9]+", "-", base).strip("-")

    candidates = []
    for candidate in (no_space, hyphenated):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


async def _probe_greenhouse(candidate: str) -> bool:
    url = f"https://boards-api.greenhouse.io/v1/boards/{candidate}/jobs"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url)
            return resp.status_code == 200
    except Exception:
        return False


async def _probe_lever(candidate: str) -> bool:
    url = f"https://api.lever.co/v0/postings/{candidate}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(url, params={"mode": "json"})
            return resp.status_code == 200
    except Exception:
        return False


async def guess_ats_slugs(company_name: str) -> dict:
    """
    Returns {"greenhouse": slug_or_None, "lever": slug_or_None}. Always
    surface this as a suggestion to confirm in the UI, never auto-save —
    a matching slug for a similarly-named but different company is possible.
    """
    candidates = _slug_candidates(company_name)

    greenhouse_hit = None
    for candidate in candidates:
        if await _probe_greenhouse(candidate):
            greenhouse_hit = candidate
            break

    lever_hit = None
    for candidate in candidates:
        if await _probe_lever(candidate):
            lever_hit = candidate
            break

    return {"greenhouse": greenhouse_hit, "lever": lever_hit}
