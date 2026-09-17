import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rss_ingest import FeedSource, FeedSourceType, build_feed_url, parse_feed  # noqa: E402


def test_build_feed_url():
    cases = [
        (FeedSource(FeedSourceType.PROFILE, "jsmith"), "https://medium.com/feed/@jsmith"),
        (FeedSource(FeedSourceType.PROFILE, "@jsmith"), "https://medium.com/feed/@jsmith"),
        (FeedSource(FeedSourceType.PUBLICATION, "example-publication"),
         "https://medium.com/feed/example-publication"),
        (FeedSource(FeedSourceType.TOPIC, "distributed-systems"),
         "https://medium.com/feed/tag/distributed-systems"),
        (FeedSource(FeedSourceType.CUSTOM_DOMAIN, "blog.example.com"),
         "https://blog.example.com/feed"),
        # Regression: a CUSTOM_DOMAIN identifier that already has a scheme
        # baked in (e.g. pasted as a full URL into the manual-entry form)
        # must not produce "https://https://..." — see WO#35 bug report.
        (FeedSource(FeedSourceType.CUSTOM_DOMAIN, "https://blog.example.com"),
         "https://blog.example.com/feed"),
        (FeedSource(FeedSourceType.CUSTOM_DOMAIN, "https://blog.example.com/"),
         "https://blog.example.com/feed"),
        (FeedSource(FeedSourceType.CUSTOM_DOMAIN, "http://blog.example.com"),
         "https://blog.example.com/feed"),
    ]
    for source, expected in cases:
        got = build_feed_url(source)
        assert got == expected, f"{source} -> {got!r}, expected {expected!r}"
    print(f"build_feed_url: OK ({len(cases)}/{len(cases)} cases)")


def test_parse_sample_feed():
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "sample_feed.xml"
    raw = fixture.read_bytes()
    source = FeedSource(FeedSourceType.PUBLICATION, "example-publication")

    articles = parse_feed(raw, source)
    assert len(articles) == 2, f"expected 2 articles, got {len(articles)}"

    for a in articles:
        assert a.guid, "guid should never be empty"
        assert a.title, "title should never be empty"
        assert a.content_html, "content_html should be pulled from content:encoded"
        assert a.published_at is not None, "pubDate should parse"
        assert a.author, "dc:creator should populate author"
        assert a.raw_item.startswith("<item"), "raw_item should hold the verbatim <item> element"
        assert a.title in a.raw_item, "raw_item should contain the same data the parsed fields came from"

    first = articles[0]
    assert first.title == "Understanding Distributed Consensus"
    assert first.tags == ["distributed-systems", "engineering"]
    assert first.published_at.year == 2025 and first.published_at.month == 9
    assert first.thumbnail_url == "https://cdn-images-1.medium.com/max/1024/0*fakecover.png", (
        "should find the figure image and skip the trailing 1x1 tracking pixel"
    )

    second = articles[1]
    assert second.thumbnail_url is None, "second fixture item has no <img> at all — must stay None, not guess"

    print("parse_feed: OK (2/2 articles)")
    for a in articles:
        print(f"  - {a.title!r} by {a.author}, tags={a.tags}, published_at={a.published_at.isoformat()}")


if __name__ == "__main__":
    test_build_feed_url()
    test_parse_sample_feed()
    print("All tests passed.")
