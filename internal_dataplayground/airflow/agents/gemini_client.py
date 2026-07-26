# airflow/agents/gemini_client.py
"""
Shared Gemini REST client for Airflow agent modules.

Why this exists:
  blog_agents.py, recipe_agents.py, and weekly_agents.py each independently
  define a private `_gemini_flash_json()`-style function that does the same
  three things: build the REST payload, enforce a JSON schema, retry on
  transient errors. They differ only in which model they call and which
  status codes they retry on (most only retry 503, not 429).

  This module is the one place that logic lives for anything new. job_agents.py
  (and job_ats_agents.py, below) call call_gemini_json() / call_gemini_text()
  instead of redefining their own copy.

  blog_agents.py, recipe_agents.py, and weekly_agents.py were intentionally
  NOT touched — they work fine as-is, and this task didn't call for editing
  three already-shipped pipelines. If you want them consolidated onto this
  client too, it's a clean, low-risk follow-up: swap their internal call
  sites for these two functions — no prompt or schema changes needed.
"""
import logging
import os
import time

import requests

log = logging.getLogger(__name__)

# Model IDs in one place — swap here rather than hunting through call sites.
MODEL_FLASH = "gemini-2.5-flash"
MODEL_FLASH_LITE = "gemini-2.5-flash-lite"

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _gemini_key() -> str:
    return os.environ.get("GEMINI_API")


def _post_with_retry(model: str, payload: dict, retries: int) -> dict:
    url = f"{_BASE_URL}/{model}:generateContent?key={_gemini_key()}"
    for attempt in range(retries):
        resp = requests.post(url, json=payload, timeout=90)

        if resp.status_code == 429:
            wait = 30 * (attempt + 1)
            log.warning("Gemini 429 on %s, waiting %ds (attempt %d/%d)",
                        model, wait, attempt + 1, retries)
            time.sleep(wait)
            continue

        if resp.status_code == 503:
            wait = 5 ** attempt
            log.warning("Gemini 503 on %s, retrying in %ds (attempt %d/%d)",
                        model, wait, attempt + 1, retries)
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(f"Gemini {model} unavailable after {retries} retries")


def call_gemini_text(system: str, prompt: str, model: str = MODEL_FLASH, retries: int = 3) -> str:
    """Free-text response, no schema enforcement — for README/Editor-style agents."""
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents":          [{"parts": [{"text": prompt}]}],
    }
    data = _post_with_retry(model, payload, retries)
    return data["candidates"][0]["content"]["parts"][0]["text"]


def call_gemini_json(
    system: str,
    prompt: str,
    schema: dict,
    model: str = MODEL_FLASH,
    retries: int = 3,
) -> str:
    """Structured JSON response. Returns the raw JSON string — caller does json.loads()."""
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents":          [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema":   schema,
        },
    }
    data = _post_with_retry(model, payload, retries)
    return data["candidates"][0]["content"]["parts"][0]["text"]
