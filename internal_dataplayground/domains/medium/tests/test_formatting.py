import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from formatting import estimate_read_minutes, relative_time  # noqa: E402


def test_estimate_read_minutes():
    assert estimate_read_minutes("") == 1
    assert estimate_read_minutes(None) == 1  # tolerate missing content_html
    assert estimate_read_minutes("<p>one two three</p>") == 1  # rounds up to a floor of 1
    long_text = "<p>" + ("word " * 400) + "</p>"  # 400 words / 200 wpm = 2.0
    assert estimate_read_minutes(long_text) == 2
    print("estimate_read_minutes: OK")


def test_relative_time():
    now = datetime(2026, 9, 13, 12, 0, 0, tzinfo=timezone.utc)
    cases = [
        (now - timedelta(seconds=10), "just now"),
        (now - timedelta(minutes=5), "5m ago"),
        (now - timedelta(hours=3), "3h ago"),
        (now - timedelta(days=2), "2d ago"),
        (now - timedelta(days=10), "1w ago"),
        (now - timedelta(days=60), "2mo ago"),
        (now - timedelta(days=400), "1y ago"),
    ]
    for dt, expected in cases:
        got = relative_time(dt, now=now)
        assert got == expected, f"{dt} -> {got!r}, expected {expected!r}"

    # naive datetime (no tzinfo) should be treated as UTC, not raise
    naive = datetime(2026, 9, 13, 6, 0, 0)
    assert relative_time(naive, now=now) == "6h ago"

    print("relative_time: OK (7/7 cases)")


if __name__ == "__main__":
    test_estimate_read_minutes()
    test_relative_time()
    print("All tests passed.")
