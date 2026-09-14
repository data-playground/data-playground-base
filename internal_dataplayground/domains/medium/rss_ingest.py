# domains/medium/rss_ingest.py
"""
Fetches and parses Medium RSS feeds (profile, publication, topic/tag,
and custom-domain publication feeds) into a storage-agnostic list of
ParsedArticle records.

WO#35 — "safe flank" of the two-flank plan: this handles publications
and profiles you already follow, via Medium's public, documented RSS
feeds. It never touches the internal GraphQL endpoint, requires no
authentication, and has no personalization — it's just what's actually
published, in reverse-chronological order, for a known source.

Deliberately built on the stdlib (xml.etree.ElementTree + email.utils)
rather than adding a feedparser dependency. Medium's feed output is a
plain, well-behaved RSS 2.0 dialect (content:encoded + dc:creator +
category) — there's no real functional gain from a heavier general-
purpose feed-parsing library for one well-known feed shape. Same
"don't add a dependency without a real functional gain" discipline
services/ai/README.md's SDK Exceptions section already applies
elsewhere in this codebase.

Storage-agnostic on purpose: this module does NOT write to the
database. Per WO#35's own boundary, any DAG built around this must not
import database.py or models.py directly — persistence belongs in a
separate dag_db.py that takes the ParsedArticle list this module
returns and handles dedup/insert (e.g. keyed on `guid`) itself.

Known limitation, not a bug: Medium's RSS feeds never include full text
for paywalled ("Member-only") stories, regardless of the requesting
account's own membership status — RSS requests aren't authenticated as
any particular Medium user. Full text for paywalled articles is a
GraphQL-with-session-cookie problem, not an RSS one; see the WO#35
Phase 1 notes on the GraphQL flank for that.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import requests

log = logging.getLogger(__name__)

_USER_AGENT = (
    "life-os-medium-ingest/0.1 "
    "(RSS reader; public/followed feeds only, no personalization or paywall access)"
)

_NS = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


class FeedSourceType(str, Enum):
    PROFILE = "profile"
    PUBLICATION = "publication"
    TOPIC = "topic"
    CUSTOM_DOMAIN = "custom_domain"


@dataclass
class FeedSource:
    """One feed to poll. `identifier` meaning depends on `type`:
      PROFILE        -> username, no leading @
      PUBLICATION    -> publication slug (medium.com/<slug>)
      TOPIC          -> tag slug (medium.com/tag/<slug>) — used for the
                        "newly discovered via GraphQL, now just follow
                        it normally" half of the two-flank plan
      CUSTOM_DOMAIN  -> full domain, e.g. "blog.example.com"
    """
    type: FeedSourceType
    identifier: str


@dataclass
class ParsedArticle:
    guid: str
    title: str
    url: str
    author: str
    published_at: datetime | None
    tags: list[str]
    content_html: str
    source_type: FeedSourceType
    source_identifier: str
    fetched_at: datetime
    raw_item: str  # the original <item> element, verbatim — see _parse_item()


def build_feed_url(source: FeedSource) -> str:
    if source.type is FeedSourceType.PROFILE:
        return f"https://medium.com/feed/@{source.identifier.lstrip('@')}"
    if source.type is FeedSourceType.PUBLICATION:
        return f"https://medium.com/feed/{source.identifier}"
    if source.type is FeedSourceType.TOPIC:
        return f"https://medium.com/feed/tag/{source.identifier}"
    if source.type is FeedSourceType.CUSTOM_DOMAIN:
        return f"https://{source.identifier.rstrip('/')}/feed"
    raise ValueError(f"Unsupported feed source type: {source.type!r}")


def fetch_feed(url: str, timeout: float = 15.0) -> bytes:
    resp = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def parse_feed(raw: bytes, source: FeedSource) -> list[ParsedArticle]:
    root = ET.fromstring(raw)
    items = root.findall("./channel/item")
    if not items:
        log.warning("No <item> entries found for %s (%s)", source.identifier, source.type.value)
    return [_parse_item(item, source) for item in items]


def _text(item: ET.Element, tag: str, ns: str | None = None) -> str:
    path = f"{{{_NS[ns]}}}{tag}" if ns else tag
    el = item.find(path)
    return (el.text or "").strip() if el is not None else ""


def _parse_item(item: ET.Element, source: FeedSource) -> ParsedArticle:
    guid = _text(item, "guid") or _text(item, "link")

    published_at = None
    pub_date_raw = _text(item, "pubDate")
    if pub_date_raw:
        try:
            published_at = parsedate_to_datetime(pub_date_raw)
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            log.debug("Unparseable pubDate %r for guid=%s", pub_date_raw, guid)

    content_html = _text(item, "encoded", ns="content") or _text(item, "description")
    tags = [el.text.strip() for el in item.findall("category") if el.text and el.text.strip()]

    # Kept verbatim specifically so a future field we didn't think to
    # extract today can be backfilled from already-stored rows later —
    # Medium's live feed only ever shows the most recent ~10-25 items
    # per source, so once an item ages out there's no going back for it.
    raw_item = ET.tostring(item, encoding="unicode")

    return ParsedArticle(
        guid=guid,
        title=_text(item, "title"),
        url=_text(item, "link"),
        author=_text(item, "creator", ns="dc") or source.identifier,
        published_at=published_at,
        tags=tags,
        content_html=content_html,
        source_type=source.type,
        source_identifier=source.identifier,
        fetched_at=datetime.now(timezone.utc),
        raw_item=raw_item,
    )


@dataclass
class DetectionResult:
    matched: bool
    source_type: FeedSourceType | None
    identifier: str | None
    sample_titles: list[str]
    attempted: list[tuple[str, str]]  # (feed_url, outcome) — for a "here's what I tried" error message


def candidate_sources(raw_input: str) -> list[tuple[FeedSourceType, str]]:
    """Given whatever a person pastes into the settings page — a plain
    Medium URL (profile, publication, topic, or a specific article's
    permalink under any of those), a bare "@handle", or a custom
    domain — returns (type, identifier) candidates worth trying, most
    likely first.

    This isn't blind guessing across all four types: the input's own
    shape already rules most of them out (a "tag/" path can only be a
    topic; a non-medium.com host can only be a custom domain), so
    there's normally exactly one candidate. The one deliberate
    exception is a bare medium.com slug with no "@" or "tag/" prefix —
    that's tried as a publication first, then as "@slug" in case the
    "@" was just left off when pasting.

    Known gap, not silently papered over: this doesn't handle every
    Medium URL shape that's ever existed (e.g. the older
    username.medium.com subdomain style some accounts still use, or
    Medium's /m/ short links). Those will fall through to CUSTOM_DOMAIN
    or return no candidates, and need manual entry.
    """
    raw_input = raw_input.strip()
    if not raw_input:
        return []

    # Bare "@handle" with no domain at all, e.g. pasted from a profile card.
    if raw_input.startswith("@") and "/" not in raw_input and "." not in raw_input:
        return [(FeedSourceType.PROFILE, raw_input)]

    url = raw_input if "://" in raw_input else f"https://{raw_input}"
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower().removeprefix("www.")
    path = parsed.path.strip("/")

    candidates: list[tuple[FeedSourceType, str]] = []

    if host in ("medium.com", ""):
        if path.startswith("@"):
            candidates.append((FeedSourceType.PROFILE, path.split("/")[0]))
        elif path.startswith("tag/"):
            tag = path[len("tag/"):].split("/")[0]
            if tag:
                candidates.append((FeedSourceType.TOPIC, tag))
        elif path:
            slug = path.split("/")[0]
            candidates.append((FeedSourceType.PUBLICATION, slug))
            candidates.append((FeedSourceType.PROFILE, f"@{slug}"))
    else:
        candidates.append((FeedSourceType.CUSTOM_DOMAIN, host))

    return candidates


def identify_source(raw_input: str) -> DetectionResult:
    """Tries each candidate build_feed_url() implies for raw_input, in
    order, and returns the first one whose feed actually returns
    articles. Used by the settings page so adding a source is "paste
    the link you'd normally visit" instead of "know which of four feed
    URL shapes applies and build it yourself."""
    attempted: list[tuple[str, str]] = []
    for source_type, identifier in candidate_sources(raw_input):
        source = FeedSource(type=source_type, identifier=identifier)
        url = build_feed_url(source)
        try:
            raw = fetch_feed(url)
            articles = parse_feed(raw, source)
        except Exception as exc:
            attempted.append((url, f"failed: {exc}"))
            continue

        if articles:
            attempted.append((url, f"OK — {len(articles)} article(s) found"))
            return DetectionResult(
                matched=True,
                source_type=source_type,
                identifier=identifier,
                sample_titles=[a.title for a in articles[:3]],
                attempted=attempted,
            )
        attempted.append((url, "reachable but returned 0 articles"))

    return DetectionResult(
        matched=False, source_type=None, identifier=None, sample_titles=[], attempted=attempted,
    )


def ingest_sources(sources: list[FeedSource]) -> list[ParsedArticle]:
    """Fetch + parse every source, skipping (and logging) any single
    feed that fails rather than aborting the whole run — one dead or
    renamed publication feed shouldn't block every other source in a
    daily run."""
    all_articles: list[ParsedArticle] = []
    for source in sources:
        url = build_feed_url(source)
        try:
            raw = fetch_feed(url)
            articles = parse_feed(raw, source)
            log.info(
                "Fetched %d article(s) from %s (%s)",
                len(articles), source.identifier, source.type.value,
            )
            all_articles.extend(articles)
        except Exception as exc:
            log.warning("Skipping %s (%s) — %s", source.identifier, url, exc)
    return all_articles
