# domains/medium/formatting.py
"""
Small, dependency-free display helpers for the articles page. Kept out
of the router/template so the actual logic (word counting, time-delta
bucketing) is testable without spinning up FastAPI or a database.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

_TAG_RE = re.compile(r"<[^>]+>")


def estimate_read_minutes(html_or_text: str, words_per_minute: int = 200) -> int:
    """Rough estimate from stripped word count. Always returns at least 1
    — a 1-sentence paywalled excerpt shouldn't display "0 min read"."""
    if not html_or_text:
        return 1
    text = _TAG_RE.sub(" ", html_or_text)
    word_count = len(text.split())
    return max(1, round(word_count / words_per_minute))


def relative_time(dt: datetime, now: datetime | None = None) -> str:
    """'3d ago' / '2h ago' style relative timestamp. Naive datetimes are
    assumed UTC (matches how rss_ingest.py and the DB store them)."""
    now = now or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    seconds = (now - dt).total_seconds()
    if seconds < 0:
        return "just now"
    if seconds < 60:
        return "just now"

    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)}m ago"

    hours = minutes / 60
    if hours < 24:
        return f"{int(hours)}h ago"

    days = hours / 24
    if days < 7:
        return f"{int(days)}d ago"

    weeks = days / 7
    if weeks < 5:
        return f"{int(weeks)}w ago"

    months = days / 30
    if months < 12:
        return f"{int(months)}mo ago"

    years = days / 365
    return f"{int(years)}y ago"
