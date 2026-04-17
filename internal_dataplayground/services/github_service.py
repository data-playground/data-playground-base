# services/github_service.py
"""
GitHub REST API service.
Handles pulling file content and pushing updated files back to the repo.
Uses a PAT stored in GCP Secret Manager under the key "GitHub-PAT".

Required GCP Secret: "GitHub-PAT"
  Value: a GitHub Personal Access Token with repo scope (read + write)
  Create at: https://github.com/settings/tokens
"""

import base64
import logging
from typing import Optional
import httpx
from database import get_key

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


def _get_headers() -> dict:
    token = get_key("GitHub-PAT")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def pull_file_content(repo: str, path: str) -> tuple[str, str]:
    """
    Fetches a single file's raw content and its current SHA from GitHub.

    Args:
        repo: "owner/repo-name", e.g. "pedro/data-playground-base"
        path: file path within the repo, e.g. "internal_dataplayground/routers/finance.py"

    Returns:
        (content: str, sha: str)
        SHA is required for pushing updates — GitHub rejects a PUT without it.

    Raises:
        httpx.HTTPStatusError if the file is not found or auth fails.
    """
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers=_get_headers())
        resp.raise_for_status()
        data = resp.json()

    content = base64.b64decode(data["content"]).decode("utf-8")
    sha = data["sha"]
    log.info("Pulled %s from %s (sha: %s)", path, repo, sha[:8])
    return content, sha


async def list_repo_files(repo: str, base_path: str = "") -> list[dict]:
    """
    Recursively lists all files under base_path in the repo.
    Returns only files (not directories), filtered to .py files by default.

    Args:
        repo: "owner/repo-name"
        base_path: folder path within repo, empty string = repo root

    Returns:
        List of {"name": str, "path": str, "sha": str}
    """
    url = f"{GITHUB_API}/repos/{repo}/git/trees/HEAD"
    params = {"recursive": "1"}

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, headers=_get_headers(), params=params)
        resp.raise_for_status()
        data = resp.json()

    files = []
    for item in data.get("tree", []):
        if item["type"] != "blob":
            continue
        path = item["path"]

        # Filter to base_path scope
        if base_path and not path.startswith(base_path.rstrip("/") + "/"):
            continue

        # Only track Python files — adjust this filter as needed
        if not path.endswith(".py"):
            continue

        files.append({
            "name": path.split("/")[-1],
            "path": path,
            "sha": item["sha"],
        })

    log.info("Found %d Python files under %s/%s", len(files), repo, base_path)
    return files


async def push_file_content(
    repo: str,
    path: str,
    content: str,
    sha: Optional[str],
    commit_message: str,
) -> str:
    """
    Creates or updates a file on GitHub.

    Args:
        repo: "owner/repo-name"
        path: file path within repo
        content: full file content as a string
        sha: current SHA of the file on GitHub (None only for brand-new files)
        commit_message: commit message string

    Returns:
        new_sha: the SHA of the file after the commit

    Raises:
        httpx.HTTPStatusError on failure (409 = SHA conflict, 422 = bad path)
    """
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")

    payload: dict = {
        "message": commit_message,
        "content": encoded,
    }
    if sha:
        payload["sha"] = sha  # Required for updates; omit only for new files

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.put(url, headers=_get_headers(), json=payload)
        resp.raise_for_status()

    new_sha = resp.json()["content"]["sha"]
    log.info("Pushed %s to %s (new sha: %s)", path, repo, new_sha[:8])
    return new_sha


async def get_file_sha(repo: str, path: str) -> Optional[str]:
    """
    Fetches just the SHA of a file without downloading its full content.
    Used to check if a README already exists before pushing.
    Returns None if the file does not exist.
    """
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, headers=_get_headers())
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()["sha"]
