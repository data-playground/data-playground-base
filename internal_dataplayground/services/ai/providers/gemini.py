# services/ai/providers/gemini.py
"""
Gemini provider implementation for the AI Service Layer (GOVERNANCE.md §2.3).

Moved from airflow/agents/gemini_client.py — MODEL_FLASH, MODEL_FLASH_LITE,
call_gemini_text(), and call_gemini_json() are unchanged in behavior. Only
two things moved out:
  - The retry/backoff loop -> services.ai.base.post_with_retry (shared
    with providers/groq.py, providers/cerebras.py).
  - The API key lookup -> services.ai.keys.get_provider_key("gemini")
    (replacing gemini_client.py's private _gemini_key()).

Function names are kept IDENTICAL to what job_agents.py already imports.
call_gemini_json()'s signature was generalized under WO#13 to make
`schema` and `system` optional keyword arguments. See call_gemini_json()'s
own docstring for the backward-compatibility note.

WO#16 adds call_gemini_vision_json() — Gemini's inlineData image +
text-prompt call shape, used by
recipe_agents.py::agent_extract_recipe_from_image(). This closes the
vision gap flagged in the WO#12 postmortem (that function previously
stayed on its own raw requests.post() call, with only its transport/
retry layer routed through post_with_retry directly, per that
postmortem's Amendment 6).
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
    enforced. When omitted, only responseMimeType: "application/json" is
    set.

    `system` is optional: when omitted, no systemInstruction is sent at
    all, rather than sending an empty one.
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


# ── WO#16: Vision support ───────────────────────────────────────────────────

def call_gemini_vision_json(
    system: str,
    image_base64: str,
    mime_type: str,
    prompt: str,
    schema: dict,
    model: str = MODEL_FLASH,
    retries: int = 3,
    timeout: float = 120.0,
) -> str:
    """
    Calls Gemini with an image (inlineData) plus a text prompt, enforcing
    JSON schema output. Used for image-based extraction — currently the
    sole caller is recipe_agents.py::agent_extract_recipe_from_image().

    Matches that function's pre-WO#16 raw payload structure exactly: a
    systemInstruction, a contents list with one inlineData part
    (mimeType + base64 data) and one text part, and a generationConfig
    with responseMimeType + responseSchema.

    `timeout` defaults to 120.0s, not the 90.0s default every other
    function in this module gets from post_with_retry — vision payloads
    (inline base64 image + text) are larger and slower. This matches
    agent_extract_recipe_from_image()'s pre-WO#16 behavior exactly, which
    passed timeout=120 explicitly (see the WO#12 postmortem's Amendment
    6). Exposed as a parameter rather than hardcoded, for any future
    non-recipe vision caller with different payload-size needs — the
    default is chosen specifically to preserve this function's one real
    caller's existing behavior unchanged, which WO#16's own acceptance
    criteria require ("identical request... to what
    agent_extract_recipe_from_image()'s original raw implementation
    produced").
    """
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{
            "parts": [
                {"inlineData": {"mimeType": mime_type, "data": image_base64}},
                {"text": prompt},
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    data = post_with_retry(
        _build_url(model), payload, retries,
        provider_name="Gemini", resource_label=model,
        timeout=timeout,
    )
    return data["candidates"][0]["content"]["parts"][0]["text"]
