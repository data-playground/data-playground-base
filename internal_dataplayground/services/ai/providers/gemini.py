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

Function names are kept IDENTICAL to what job_agents.py already imports.
call_gemini_json()'s signature was generalized under WO#13 to make
`schema` and `system` optional keyword arguments (was previously
positional `(system, prompt, schema, model=..., retries=...)`) so it
could also serve callers with no schema (workout_plan_ai_generator.py,
media_recommend.py) or no system instruction (media_recommend.py). See
call_gemini_json()'s own docstring for the backward-compatibility note.
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
    prompt: str,
    schema: dict | None = None,
    system: str | None = None,
    model: str = MODEL_FLASH,
    retries: int = 3,
) -> str:
    """
    Structured JSON response. Returns the raw JSON string — caller does json.loads().

    `schema` is optional: when provided, a Gemini responseSchema is
    enforced (OBJECT/ARRAY-shaped structured output). When omitted, only
    responseMimeType: "application/json" is set — free-form JSON, caller
    validates shape itself (e.g. media_recommend.py's _gemini_explain()).

    `system` is optional: when omitted, no systemInstruction is sent at
    all, rather than sending an empty one (media_recommend.py's
    _gemini_explain() relies on this — the absence of a system
    instruction there is intentional, not an oversight).

    NOTE ON BACKWARD COMPATIBILITY (WO#13): this signature reorders and
    changes the defaults of the original (system, prompt, schema, model,
    retries) signature used by job_agents.py (WO#11) and recipe_agents.py
    (WO#12). Both files' call sites already relied on positional order
    and would silently pass arguments to the wrong parameters under the
    new signature — they've been updated to explicit keyword arguments
    as part of this change (see job_agents.py / recipe_agents.py diffs).
    """
    generation_config = {"responseMimeType": "application/json"}
    if schema is not None:
        generation_config["responseSchema"] = schema

    payload = {
        "contents":          [{"parts": [{"text": prompt}]}],
        "generationConfig":  generation_config,
    }
    if system is not None:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    data = post_with_retry(
        _build_url(model), payload, retries,
        provider_name="Gemini", resource_label=model,
    )
    return data["candidates"][0]["content"]["parts"][0]["text"]
