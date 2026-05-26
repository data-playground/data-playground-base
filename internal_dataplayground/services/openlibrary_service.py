# services/openlibrary_service.py
"""
OpenLibrary API service.

No API key required. All calls use the public OpenLibrary REST API.

Base URL: https://openlibrary.org
Cover images: https://covers.openlibrary.org/b/id/{cover_id}-M.jpg

Key design decisions:
  - Search uses /search.json which is OpenLibrary's full-text search endpoint.
    It returns works (not editions) so results are deduplicated across editions.
  - external_id stores the OpenLibrary work ID (e.g. "/works/OL45883W").
    This is the stable canonical identifier — edition IDs change, work IDs don't.
  - Cover images use the cover_i field from search results, which is a numeric
    cover ID. We construct the URL directly from it.
  - Subjects (genres) from OpenLibrary are noisy — we take only the first 5
    and capitalize them. Genre matching for recommendations uses fuzzy matching.
  - Page count (number_of_pages_median) is from OpenLibrary's aggregated data
    across editions — it's a median, so it may be imprecise.
  - Rating data from OpenLibrary is sparse and unreliable; we store it if
    present but don't rely on it for recommendations.
"""

import logging
from typing import Optional
import httpx

log = logging.getLogger(__name__)

OL_BASE = "https://openlibrary.org"
OL_COVER_BASE = "https://covers.openlibrary.org/b/id"

# OpenLibrary blocks generic user agents — identify your app clearly
_OL_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


def _build_cover_url(cover_id: Optional[int], size: str = "M") -> Optional[str]:
    if not cover_id:
        return None
    return f"{OL_COVER_BASE}/{cover_id}-{size}.jpg"


def _clean_subjects(subjects: list) -> list[str]:
    seen = set()
    result = []
    for s in subjects:
        s_clean = str(s).strip().title()
        if len(s_clean) > 40 or s_clean.lower() in seen:
            continue
        seen.add(s_clean.lower())
        result.append(s_clean)
        if len(result) >= 5:
            break
    return result


async def search_books(query: str, author: str = "", limit: int = 10) -> list[dict]:
    params = {
        "q": query,
        "fields": "key,title,author_name,first_publish_year,subject,"
                  "cover_i,number_of_pages_median,ratings_average",
        "limit": limit,
    }
    if author:
        params["author"] = author

    async with httpx.AsyncClient(timeout=15.0, headers=_OL_HEADERS) as client:
        resp = await client.get(f"{OL_BASE}/search.json", params=params)
        resp.raise_for_status()
        data = resp.json()

    results = []
    for item in data.get("docs", []):
        work_key = item.get("key", "")
        if not work_key:
            continue

        authors = item.get("author_name", [])
        author_str = ", ".join(authors[:2]) if authors else None

        subjects = _clean_subjects(item.get("subject", []))

        rating = item.get("ratings_average")
        try:
            rating = round(float(rating), 1) if rating else None
        except (TypeError, ValueError):
            rating = None

        results.append({
            "external_id":     work_key,
            "external_source": "openlibrary",
            "title":           item.get("title", "Unknown"),
            "media_type":      "book",
            "genres":          subjects,
            "release_year":    item.get("first_publish_year") or None,
            "description":     None,
            "poster_url":      _build_cover_url(item.get("cover_i")),
            "external_rating": rating,
            "author":          author_str,
            "page_count":      item.get("number_of_pages_median") or None,
        })
    return results


async def get_book_details(work_key: str) -> dict:
    key = work_key.lstrip("/")
    async with httpx.AsyncClient(timeout=15.0, headers=_OL_HEADERS) as client:
        resp = await client.get(f"{OL_BASE}/{key}.json")
        if resp.status_code == 404:
            return {
                "external_id": work_key, "external_source": "openlibrary",
                "title": "Unknown", "media_type": "book",
            }
        resp.raise_for_status()
        data = resp.json()

    raw_desc = data.get("description", "")
    if isinstance(raw_desc, dict):
        description = raw_desc.get("value", "")
    else:
        description = str(raw_desc)
    description = description[:2000] if description else None

    subjects = _clean_subjects(data.get("subjects", []))

    return {
        "external_id":     work_key,
        "external_source": "openlibrary",
        "title":           data.get("title", "Unknown"),
        "media_type":      "book",
        "genres":          subjects,
        "description":     description,
    }


async def get_author_name(author_key: str) -> Optional[str]:
    key = author_key.lstrip("/")
    async with httpx.AsyncClient(timeout=10.0, headers=_OL_HEADERS) as client:
        resp = await client.get(f"{OL_BASE}/{key}.json")
        if resp.status_code != 200:
            return None
        return resp.json().get("name")