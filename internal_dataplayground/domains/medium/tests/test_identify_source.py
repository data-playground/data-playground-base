import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rss_ingest  # noqa: E402
from rss_ingest import FeedSourceType, candidate_sources, identify_source  # noqa: E402


def test_candidate_sources():
    cases = [
        ("https://medium.com/@jsmith", [(FeedSourceType.PROFILE, "@jsmith")]),
        ("medium.com/@jsmith", [(FeedSourceType.PROFILE, "@jsmith")]),
        ("https://medium.com/@jsmith/some-article-title-abc123", [(FeedSourceType.PROFILE, "@jsmith")]),
        ("@jsmith", [(FeedSourceType.PROFILE, "@jsmith")]),  # bare handle, no URL at all
        ("https://medium.com/tag/distributed-systems", [(FeedSourceType.TOPIC, "distributed-systems")]),
        (
            "https://medium.com/example-publication",
            [(FeedSourceType.PUBLICATION, "example-publication"), (FeedSourceType.PROFILE, "@example-publication")],
        ),
        (
            "https://medium.com/example-publication/some-article-abc123",
            [(FeedSourceType.PUBLICATION, "example-publication"), (FeedSourceType.PROFILE, "@example-publication")],
        ),
        (
            "https://www.medium.com/example-publication",
            [(FeedSourceType.PUBLICATION, "example-publication"), (FeedSourceType.PROFILE, "@example-publication")],
        ),
        ("https://blog.example.com", [(FeedSourceType.CUSTOM_DOMAIN, "blog.example.com")]),
        ("https://www.blog.example.com/", [(FeedSourceType.CUSTOM_DOMAIN, "blog.example.com")]),
        ("blog.example.com/some-post", [(FeedSourceType.CUSTOM_DOMAIN, "blog.example.com")]),
        ("", []),
        ("   ", []),
    ]
    for raw_input, expected in cases:
        got = candidate_sources(raw_input)
        assert got == expected, f"{raw_input!r} -> {got!r}, expected {expected!r}"
    print(f"candidate_sources: OK ({len(cases)}/{len(cases)} cases)")


def test_identify_source_picks_first_working_candidate(monkeypatch):
    # example-publication yields two candidates: PUBLICATION first, then
    # PROFILE as a fallback. Only the PROFILE feed "exists" here — this
    # confirms identify_source() actually walks the list instead of
    # stopping blindly at the first guess.
    working_url = "https://medium.com/feed/@example-publication"

    def fake_fetch_feed(url, timeout=15.0):
        if url == working_url:
            return SAMPLE_FEED
        raise rss_ingest.requests.exceptions.RequestException("404 not found")

    monkeypatch.setattr(rss_ingest, "fetch_feed", fake_fetch_feed)

    result = identify_source("https://medium.com/example-publication")
    assert result.matched is True
    assert result.source_type is FeedSourceType.PROFILE
    assert result.identifier == "@example-publication"
    assert len(result.sample_titles) == 2
    assert len(result.attempted) == 2  # tried PUBLICATION (failed), then PROFILE (succeeded)
    print("identify_source: picks first working candidate — OK")


def test_identify_source_no_match():
    def always_fails(url, timeout=15.0):
        raise rss_ingest.requests.exceptions.RequestException("404 not found")

    original = rss_ingest.fetch_feed
    rss_ingest.fetch_feed = always_fails
    try:
        result = identify_source("https://medium.com/some-nonexistent-thing")
        assert result.matched is False
        assert result.source_type is None
        assert len(result.attempted) == 2  # PUBLICATION + PROFILE fallback, both failed
    finally:
        rss_ingest.fetch_feed = original
    print("identify_source: reports no-match cleanly — OK")


SAMPLE_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>Example</title>
    <item>
      <title>First fake article</title>
      <link>https://medium.com/example-publication/first-fake-article</link>
      <guid isPermaLink="false">https://medium.com/p/aaa111</guid>
      <pubDate>Fri, 12 Sep 2025 14:30:00 GMT</pubDate>
      <dc:creator>Test Author</dc:creator>
      <content:encoded><![CDATA[<p>placeholder</p>]]></content:encoded>
    </item>
    <item>
      <title>Second fake article</title>
      <link>https://medium.com/example-publication/second-fake-article</link>
      <guid isPermaLink="false">https://medium.com/p/bbb222</guid>
      <pubDate>Wed, 10 Sep 2025 09:15:00 GMT</pubDate>
      <dc:creator>Test Author</dc:creator>
      <content:encoded><![CDATA[<p>placeholder</p>]]></content:encoded>
    </item>
  </channel>
</rss>
"""


class _FakeMonkeypatch:
    """Minimal stand-in for pytest's monkeypatch fixture so these tests
    run with plain `python3 test_identify_source.py` — no pytest
    dependency needed for a two-function test file."""

    def __init__(self):
        self._restore = []

    def setattr(self, obj, name, value):
        self._restore.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old_value in self._restore:
            setattr(obj, name, old_value)


if __name__ == "__main__":
    test_candidate_sources()

    mp = _FakeMonkeypatch()
    try:
        test_identify_source_picks_first_working_candidate(mp)
    finally:
        mp.undo()

    test_identify_source_no_match()
    print("All tests passed.")
