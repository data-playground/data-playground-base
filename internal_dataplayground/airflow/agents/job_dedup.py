# airflow/agents/job_dedup.py
"""
Cross-source duplicate detection.

Once ATS sources (Greenhouse/Lever) run alongside the LinkedIn scraper, the
exact same role can legitimately appear from two sources — a company posts
to LinkedIn AND their own board. The (source, external_ref) uniqueness
constraint only prevents re-inserting the SAME posting from the SAME source
twice; it does nothing for "this is a different row representing the same
real job."

This does lightweight identity matching: normalized company name + fuzzy
title match against jobs already in linkedin_jobs from a DIFFERENT source,
within a recency window. It intentionally does NOT try to catch every
duplicate — a missed duplicate just costs one extra card; a false-positive
match means a real, distinct job silently never shows up, which is worse.
The threshold below is tuned conservative for that reason.
"""
import logging
import re
from difflib import SequenceMatcher

log = logging.getLogger(__name__)

TITLE_SIMILARITY_THRESHOLD = 0.87
RECENCY_WINDOW_DAYS = 45  # callers should pre-filter existing_rows to this window


def _normalize(text: str | None) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\b(inc|llc|corp|corporation|co|ltd)\.?\b", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _titles_match(a: str, b: str) -> bool:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio() >= TITLE_SIMILARITY_THRESHOLD


def filter_cross_source_duplicates(
    new_jobs: list[dict],
    existing_rows: list[dict],
) -> tuple[list[dict], int]:
    """
    Args:
        new_jobs:      Jobs about to be inserted (any source). Each needs
                        job_title, company_name, and source.
        existing_rows: Recent rows already in linkedin_jobs — each dict
                       needs company_name, job_title, source. Callers should
                       pre-filter this to RECENCY_WINDOW_DAYS to keep the
                       O(n*m) comparison cheap (typically a few hundred rows
                       at most, comparing against a batch of a few dozen).

    Returns:
        (deduped_jobs, skipped_count)
    """
    by_company: dict[str, list[dict]] = {}
    for row in existing_rows:
        key = _normalize(row.get("company_name"))
        if key:
            by_company.setdefault(key, []).append(row)

    deduped = []
    skipped = 0
    for job in new_jobs:
        company_key = _normalize(job.get("company_name"))
        candidates = by_company.get(company_key, [])
        is_dup = any(
            existing["source"] != job.get("source", "linkedin")
            and _titles_match(existing["job_title"], job["job_title"])
            for existing in candidates
        )
        if is_dup:
            skipped += 1
            log.info(
                "Skipping likely cross-source duplicate: '%s' @ %s (source=%s)",
                job.get("job_title"), job.get("company_name"), job.get("source", "linkedin"),
            )
        else:
            deduped.append(job)

    return deduped, skipped
