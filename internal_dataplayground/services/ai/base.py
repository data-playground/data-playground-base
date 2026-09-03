# services/ai/base.py
#
# RECONSTRUCTED STUB — not provided in the source documents for this work
# order. Inferred purely from call-site usage:
#   - services/ai/providers/gemini.py:
#       post_with_retry(url, payload, retries, provider_name=..., resource_label=...)
#   - airflow/agents/recipe_agents.py (agent_extract_recipe_from_image):
#       post_with_retry(url, payload, retries=3, provider_name="Gemini",
#                        resource_label=MODEL_FLASH, timeout=120)
# Signature and 429/503 handling shape is modeled on the existing bespoke
# retry loops already in this codebase (blog_agents.py's commented-out
# _cerebras() and job_agents.py's own patterns) since no other source of
# truth was available. VERIFY AGAINST THE REAL FILE before merging — do
# not treat this as ground truth.
import logging
import time

import requests

log = logging.getLogger(__name__)

# Same conservative default backoff already used elsewhere in this
# codebase for Gemini 503s (blog_agents.py's old _gemini_flash_json):
# 1s, 5s, 25s.
_DEFAULT_BACKOFF = [1, 5, 25]


def post_with_retry(
    url: str,
    payload: dict,
    retries: int = 3,
    provider_name: str = "",
    resource_label: str = "",
    timeout: float = 90.0,
) -> dict:
    """
    Shared POST-with-retry used by every services/ai/providers/*.py
    implementation. Retries on 503 (Service Unavailable) with exponential
    backoff; raises immediately on any other non-2xx status.

    Returns the parsed JSON response body. Callers index into it
    (e.g. data["candidates"][0]["content"]["parts"][0]["text"]).
    """
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            last_exc = exc
            if status == 503 and attempt < retries - 1:
                wait = _DEFAULT_BACKOFF[min(attempt, len(_DEFAULT_BACKOFF) - 1)]
                log.warning(
                    "%s %s returned 503, retrying in %ds (attempt %d/%d)",
                    provider_name, resource_label, wait, attempt + 1, retries,
                )
                time.sleep(wait)
                continue
            raise
        except Exception as exc:
            last_exc = exc
            raise

    raise RuntimeError(
        f"{provider_name} {resource_label} unavailable after {retries} retries. "
        f"Last error: {last_exc}"
    )
