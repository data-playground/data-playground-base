# services/ai/providers/gemini.py
"""
Gemini provider implementation for the AI Service Layer (GOVERNANCE.md §2.3).

Moved from airflow/agents/gemini_client.py — MODEL_FLASH, MODEL_FLASH_LITE,
call_gemini_text(), and call_gemini_json() are unchanged in behavior. Only
two things moved out:
  - The retry/backoff loop -> services.ai.base.post_with_retry (shared
    with future providers/groq.py, providers/cerebras.py).
  - The API key lookup -> services.ai.keys.get_provider_key("gemini")
    (replacing gemini_client.py's private _gemini_key()).

Function names and signatures are kept IDENTICAL to what job_agents.py
already imports and calls (`call_gemini_json(system, prompt, schema,
model=..., retries=...)`), so migrating job_agents.py is a one-line import
change rather than a call-site rewrite.
"""
import logging

from services.ai.base import post_with_retry
from services.ai.keys import get_provider_key

log = logging.getLogger(__name__)

# Model IDs in one place — swap here rather than hunting through call sites.
MODEL_FLASH = "gemini-2.5-flash"
MODEL_FLASH_LITE = "gemini-2.5-flash-lite"
MODEL_GEMMA = "gemma-4-31b-it"

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _build_url(model: str) -> str:
    """
    Same URL shape gemini_client.py built: base + model + :generateContent
    + the API key as a query param. Built once per call (not once per
    retry attempt) — matches the original, which computed `url` once
    before entering its retry loop.
    """
    return f"{_BASE_URL}/{model}:generateContent?key={get_provider_key('gemini')}"


def call_gemini_text(system: str, prompt: str, model: str = MODEL_FLASH, retries: int = 3) -> str:
    """Free-text response, no schema enforcement — for README/Editor-style agents."""
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents":          [{"parts": [{"text": prompt}]}],
    }
    data = post_with_retry(
        _build_url(model), payload, retries,
        provider_name="Gemini", resource_label=model,
    )
    return data["candidates"][0]["content"]["parts"][0]["text"]


def call_gemma_json(prompt: str, model: str = MODEL_GEMMA, retries: int = 3) -> str:
    """
    Calls a Gemma model via the Gemini API endpoint. Unlike
    call_gemini_json(), Gemma models don't support systemInstruction —
    callers must prepend any system context directly into `prompt`
    themselves, same as recipe_agents.py's original _gemma() required.
    No responseSchema enforcement — only responseMimeType: application/json.

    `retries` defaults to 3 for parity with call_gemini_text/call_gemini_json
    (the original recipe_agents.py _gemma() had no retry logic at all —
    see the WO#12 postmortem for the retry-semantics discussion this
    introduced).
    """
    payload = {
        "contents":         [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    data = post_with_retry(
        _build_url(model), payload, retries,
        provider_name="Gemini", resource_label=model,
    )
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
    data = post_with_retry(
        _build_url(model), payload, retries,
        provider_name="Gemini", resource_label=model,
    )
    return data["candidates"][0]["content"]["parts"][0]["text"]
