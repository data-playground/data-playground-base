# services/tmdb_service.py
"""
TMDB (The Movie Database) REST API service.

Handles fetching movie and TV show metadata and streaming availability.
API key stored in GCP Secret Manager as "TMDB-API-Key".

TMDB API reference: https://developers.themoviedb.org/3
Image base URL: https://image.tmdb.org/t/p/w500

Key design decisions:
  - All calls use async httpx for consistency with the rest of the app.
  - Streaming providers are fetched for the US region ("US") by default.
    This is a personal app — the user is in New York.
  - TMDB vote_average is on a 0-10 scale — stored directly as external_rating.
  - poster_path from TMDB is a relative path; we construct the full URL here
    so callers never need to know the base URL.
  - Genre names are fetched from /genre/{type}/list and cached in memory for
    the process lifetime (genres change rarely — a restart is acceptable).
"""

import logging
from typing import Optional
import httpx
from gcp_secrets import get_key

log = logging.getLogger(__name__)

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# In-memory genre cache: {"movie": {28: "Action", ...}, "tv": {18: "Drama", ...}}
_genre_cache: dict[str, dict[int, str]] = {}


def _get_api_key() -> str:
    return get_key("TMDB-API-Key")


def _build_poster_url(poster_path: Optional[str]) -> Optional[str]:
    """Constructs the full TMDB image URL from a relative poster path."""
    if not poster_path:
        return None
    return f"{TMDB_IMAGE_BASE}{poster_path}"


async def _get_headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Accept": "application/json",
    }


async def _fetch_genres(media_type: str) -> dict[int, str]:
    """
    Fetches and caches genre ID → name mapping from TMDB.
    media_type: "movie" or "tv"
    """
    global _genre_cache
    if media_type in _genre_cache:
        return _genre_cache[media_type]

    endpoint = "movie" if media_type == "movie" else "tv"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{TMDB_BASE}/genre/{endpoint}/list",
            headers=await _get_headers(),
        )
        resp.raise_for_status()
        genres = {g["id"]: g["name"] for g in resp.json().get("genres", [])}
        _genre_cache[media_type] = genres
        return genres


async def search_movies(query: str, page: int = 1) -> list[dict]:
    """
    Searches TMDB for movies matching the query string.

    Returns a list of normalized movie dicts ready for media_items insertion.
    Posters are resolved to full URLs. Genres are resolved to name strings.
    Results with no overview or no poster are included — callers can filter.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{TMDB_BASE}/search/movie",
            headers=await _get_headers(),
            params={"query": query, "page": page, "include_adult": False},
        )
        resp.raise_for_status()
        data = resp.json()

    genre_map = await _fetch_genres("movie")
    results = []
    for item in data.get("results", [])[:10]:
        year = None
        if item.get("release_date") and len(item["release_date"]) >= 4:
            try:
                year = int(item["release_date"][:4])
            except ValueError:
                pass

        results.append({
            "external_id":     str(item["id"]),
            "external_source": "tmdb_movie",
            "title":           item.get("title", "Unknown"),
            "media_type":      "movie",
            "genres":          [genre_map.get(gid, str(gid)) for gid in item.get("genre_ids", [])],
            "release_year":    year,
            "description":     item.get("overview") or None,
            "poster_url":      _build_poster_url(item.get("poster_path")),
            "external_rating": item.get("vote_average") or None,
        })
    return results


async def search_tv(query: str, page: int = 1) -> list[dict]:
    """
    Searches TMDB for TV shows matching the query string.
    Returns normalized dicts for media_items insertion.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{TMDB_BASE}/search/tv",
            headers=await _get_headers(),
            params={"query": query, "page": page},
        )
        resp.raise_for_status()
        data = resp.json()

    genre_map = await _fetch_genres("tv")
    results = []
    for item in data.get("results", [])[:10]:
        year = None
        if item.get("first_air_date") and len(item["first_air_date"]) >= 4:
            try:
                year = int(item["first_air_date"][:4])
            except ValueError:
                pass

        results.append({
            "external_id":     str(item["id"]),
            "external_source": "tmdb_tv",
            "title":           item.get("name", "Unknown"),
            "media_type":      "tv_show",
            "genres":          [genre_map.get(gid, str(gid)) for gid in item.get("genre_ids", [])],
            "release_year":    year,
            "description":     item.get("overview") or None,
            "poster_url":      _build_poster_url(item.get("poster_path")),
            "external_rating": item.get("vote_average") or None,
        })
    return results


async def get_movie_details(tmdb_id: str) -> dict:
    """
    Fetches full movie details from TMDB including runtime.
    Used when creating a media_item record for the first time.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{TMDB_BASE}/movie/{tmdb_id}",
            headers=await _get_headers(),
            params={"append_to_response": ""},
        )
        resp.raise_for_status()
        item = resp.json()

    year = None
    if item.get("release_date") and len(item["release_date"]) >= 4:
        try:
            year = int(item["release_date"][:4])
        except ValueError:
            pass

    return {
        "external_id":      str(item["id"]),
        "external_source":  "tmdb_movie",
        "title":            item.get("title", "Unknown"),
        "media_type":       "movie",
        "genres":           [g["name"] for g in item.get("genres", [])],
        "release_year":     year,
        "description":      item.get("overview") or None,
        "poster_url":       _build_poster_url(item.get("poster_path")),
        "external_rating":  item.get("vote_average") or None,
        "runtime_minutes":  item.get("runtime") or None,
    }


async def get_tv_details(tmdb_id: str) -> dict:
    """
    Fetches full TV show details from TMDB including season/episode counts.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{TMDB_BASE}/tv/{tmdb_id}",
            headers=await _get_headers(),
        )
        resp.raise_for_status()
        item = resp.json()

    year = None
    if item.get("first_air_date") and len(item["first_air_date"]) >= 4:
        try:
            year = int(item["first_air_date"][:4])
        except ValueError:
            pass

    # Sum episode counts from all season objects (excludes specials season 0)
    total_episodes = sum(
        s.get("episode_count", 0)
        for s in item.get("seasons", [])
        if s.get("season_number", 0) > 0
    )

    return {
        "external_id":    str(item["id"]),
        "external_source":"tmdb_tv",
        "title":          item.get("name", "Unknown"),
        "media_type":     "tv_show",
        "genres":         [g["name"] for g in item.get("genres", [])],
        "release_year":   year,
        "description":    item.get("overview") or None,
        "poster_url":     _build_poster_url(item.get("poster_path")),
        "external_rating":item.get("vote_average") or None,
        "total_seasons":  item.get("number_of_seasons") or None,
        "total_episodes": total_episodes or None,
    }


async def get_streaming_providers(
    tmdb_id: str,
    media_type: str,
    region: str = "US",
) -> list[int]:
    """
    Fetches streaming provider IDs for a movie or TV show in the given region.

    Args:
        tmdb_id:    TMDB item ID as string.
        media_type: "movie" or "tv_show" (we map tv_show → "tv" for the URL).
        region:     ISO 3166-1 alpha-2 country code. Default "US".

    Returns:
        List of TMDB provider IDs where the item is available to stream
        (flatrate only — excludes rent/buy). Empty list if unavailable.

    TMDB's flatrate result means subscription streaming (Netflix, Hulu, etc.).
    We intentionally exclude rent/buy to keep recommendations focused on
    what the user can watch without additional cost.
    """
    url_type = "tv" if media_type == "tv_show" else "movie"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{TMDB_BASE}/{url_type}/{tmdb_id}/watch/providers",
            headers=await _get_headers(),
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()

    region_data = data.get("results", {}).get(region, {})
    flatrate = region_data.get("flatrate", [])
    return [p["provider_id"] for p in flatrate]


async def fetch_all_provider_logos() -> dict[int, str]:
    """
    Fetches all available streaming providers from TMDB with their logo URLs.
    Used to populate the streaming_services logo_url column.
    Returns: {provider_id: logo_url}
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{TMDB_BASE}/watch/providers/movie",
            headers=await _get_headers(),
            params={"watch_region": "US"},
        )
        resp.raise_for_status()
        providers = resp.json().get("results", [])

    return {
        p["provider_id"]: _build_poster_url(p.get("logo_path"))
        for p in providers
        if p.get("logo_path")
    }
