import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rss_ingest import extract_thumbnail  # noqa: E402

_TRACKING_PIXEL = (
    '<img src="https://medium.com/_/stat?event=post.clientViewed&referrerSource=full_rss&postId=abc123" '
    'width="1" height="1" alt="">'
)


def test_image_as_first_element():
    # The common case — confirmed 6/10 in the real feed sample.
    html = '<figure><img alt="" src="https://cdn.example.com/cover.png" /></figure><p>Body text.</p>'
    assert extract_thumbnail(html) == "https://cdn.example.com/cover.png"


def test_image_after_one_paragraph():
    # Confirmed real pattern — several articles open with a lead
    # paragraph before their figure. "First child" would miss this;
    # "first <img> anywhere" does not.
    html = (
        "<p>This piece walks through a deployment guide before anything else.</p>"
        '<figure><img alt="" src="https://cdn.example.com/cover2.jpg" /></figure>'
        "<p>More body text.</p>"
    )
    assert extract_thumbnail(html) == "https://cdn.example.com/cover2.jpg"


def test_image_deep_after_heading_and_paragraphs():
    # Confirmed real pattern — one article had a byline paragraph, an
    # "Overview" heading, and a full paragraph before its image.
    html = (
        "<p>Authors: A &amp; B</p><h3>Overview</h3>"
        "<p>Paragraph one of the introduction.</p>"
        "<p>Paragraph two of the introduction.</p>"
        '<figure><img alt="" src="https://cdn.example.com/deep.png" /></figure>'
    )
    assert extract_thumbnail(html) == "https://cdn.example.com/deep.png"


def test_skips_tracking_pixel_when_real_image_present():
    html = (
        '<figure><img alt="" src="https://cdn.example.com/real-cover.png" /></figure>'
        "<p>Body text.</p>" + _TRACKING_PIXEL
    )
    assert extract_thumbnail(html) == "https://cdn.example.com/real-cover.png"


def test_tracking_pixel_only_returns_none():
    # A text-only article — no real content image, only the trailing
    # tracking pixel. Must not mistake the pixel for a thumbnail.
    html = "<p>Text-only article, no images at all.</p>" + _TRACKING_PIXEL
    assert extract_thumbnail(html) is None


def test_no_images_at_all_returns_none():
    assert extract_thumbnail("<p>Just a paragraph, nothing else.</p>") is None


def test_empty_content_returns_none():
    assert extract_thumbnail("") is None
    assert extract_thumbnail(None) is None


if __name__ == "__main__":
    test_image_as_first_element()
    test_image_after_one_paragraph()
    test_image_deep_after_heading_and_paragraphs()
    test_skips_tracking_pixel_when_real_image_present()
    test_tracking_pixel_only_returns_none()
    test_no_images_at_all_returns_none()
    test_empty_content_returns_none()
    print("All 7 extract_thumbnail tests passed.")
