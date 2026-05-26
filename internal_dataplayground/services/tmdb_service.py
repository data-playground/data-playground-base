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

_genre_cache: dict[str, dict[int, str]] = {}


def _get_api_key() -> str:
    return get_key("TMDB-API-Key")


def _build_poster_url(poster_path: Optional[str]) -> Optional[str]:
    if not poster_path:
        return None
    return f"{TMDB_IMAGE_BASE}{poster_path}"


def _base_params(extra: dict = None) -> dict:
    """All TMDB v3 calls use api_key as a query param."""
    params = {"api_key": _get_api_key()}
    if extra:
        params.update(extra)
    return params


async def _fetch_genres(media_type: str) -> dict[int, str]:
    global _genre_cache
    if media_type in _genre_cache:
        return _genre_cache[media_type]
    endpoint = "movie" if media_type == "movie" else "tv"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{TMDB_BASE}/genre/{endpoint}/list",
            params=_base_params(),
        )
        resp.raise_for_status()
        genres = {g["id"]: g["name"] for g in resp.json().get("genres", [])}
        _genre_cache[media_type] = genres
        return genres


async def search_movies(query: str, page: int = 1) -> list[dict]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{TMDB_BASE}/search/movie",
            params=_base_params({"query": query, "page": page, "include_adult": False}),
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
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{TMDB_BASE}/search/tv",
            params=_base_params({"query": query, "page": page}),
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
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{TMDB_BASE}/movie/{tmdb_id}",
            params=_base_params(),
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
    Fetches full TV show details including per-season episode counts.

    seasons_data is a dict of {season_number_str: episode_count} for all
    regular seasons (excludes season 0 / specials).
    Example: {"1": 22, "2": 22, "3": 23, "4": 14}
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{TMDB_BASE}/tv/{tmdb_id}",
            params=_base_params(),
        )
        resp.raise_for_status()
        item = resp.json()

    year = None
    if item.get("first_air_date") and len(item["first_air_date"]) >= 4:
        try:
            year = int(item["first_air_date"][:4])
        except ValueError:
            pass

    # Build per-season episode counts — exclude season 0 (specials)
    seasons_data = {}
    total_episodes = 0
    for s in item.get("seasons", []):
        snum = s.get("season_number", 0)
        ep_count = s.get("episode_count", 0)
        if snum > 0:
            seasons_data[str(snum)] = ep_count
            total_episodes += ep_count

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
        # Per-season breakdown — stored in media_items.seasons_data
        "seasons_data":   seasons_data if seasons_data else None,
    }


async def get_streaming_providers(
    tmdb_id: str,
    media_type: str,
    region: str = "US",
) -> list[int]:
    url_type = "tv" if media_type == "tv_show" else "movie"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{TMDB_BASE}/{url_type}/{tmdb_id}/watch/providers",
            params=_base_params(),
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()

    region_data = data.get("results", {}).get(region, {})
    flatrate = region_data.get("flatrate", [])
    return [p["provider_id"] for p in flatrate]


async def fetch_all_provider_logos() -> dict[int, str]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{TMDB_BASE}/watch/providers/movie",
            params=_base_params({"watch_region": "US"}),
        )
        resp.raise_for_status()
        providers = resp.json().get("results", [])

    return {
        p["provider_id"]: _build_poster_url(p.get("logo_path"))
        for p in providers
        if p.get("logo_path")
    }
