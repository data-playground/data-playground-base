# services/ai/base.py
"""
Shared low-level HTTP-call machinery for services/ai/ provider modules.

Extracted from airflow/agents/gemini_client.py's `_post_with_retry()` — the
cleanest of the six duplicate AI-call implementations in the repo (see
GOVERNANCE.md §2.3) and the one this work order migrates first. Behavior
is preserved exactly: same retry count semantics, same 429/503 wait
schedule, same eventual RuntimeError text shape. Only the shape has
changed — from a Gemini-specific helper (keyed on `model`) to a
provider-agnostic one (keyed on `provider_name` + `resource_label`) that
providers/gemini.py builds on, and that future providers/groq.py /
providers/cerebras.py modules can build on too without rework.

NOT migrated here: Groq's and Cerebras's retry/backoff logic.
  - Groq's current caller (_groq_llama in blog_agents.py) doesn't retry
    at all today.
  - Cerebras's (_cerebras in blog_agents.py) has materially different
    behavior: it honors a Retry-After header, uses a different backoff
    schedule (75/150/300/600s), and returns a (content, remaining_tokens)
    tuple rather than just content.
  Forcing either into this function now would mean guessing at what
  should be configurable before a second real provider exists in
  services/ai/providers/ to generalize from correctly. That's explicitly
  deferred — see this work order's "For the next work order" section.
"""

import logging
import time

import requests

log = logging.getLogger(__name__)


def post_with_retry(
    url: str,
    payload: dict,
    retries: int,
    *,
    provider_name: str,
    resource_label: str,
    timeout: float = 90.0,
) -> dict:
    """
    POSTs `payload` (JSON) to `url` with retry on 429 and 503, matching
    gemini_client.py's original `_post_with_retry()` exactly:

      - 429: wait 30 * (attempt + 1) seconds — 30s, 60s, 90s, ...
      - 503: wait 5 ** attempt seconds — 1s, 5s, 25s, ...
      - Any other non-2xx: resp.raise_for_status() raised immediately,
        no retry.
      - Once `retries` attempts are exhausted without a 2xx response,
        raises RuntimeError.

    `provider_name` and `resource_label` exist purely so callers can
    reproduce their original log/error text exactly (e.g. gemini.py
    passes provider_name="Gemini", resource_label=<model id>, which
    reproduces "Gemini 429 on gemini-2.5-flash, waiting 30s (attempt
    1/3)" and "Gemini gemini-2.5-flash unavailable after 3 retries"
    verbatim). This function itself has no Gemini-specific knowledge.

    Args:
        url:            Full request URL. Provider modules build this,
                         including any query-string API key — this
                         function doesn't know about auth.
        payload:         JSON-serializable request body.
        retries:         Number of attempts before giving up.
        provider_name:   Short label for log/error messages, e.g. "Gemini".
        resource_label:  Short label for log/error messages, e.g. the
                         model id being called.
        timeout:         Per-request timeout in seconds, passed straight
                         through to requests.post(). Defaults to 90.0 —
                         the value every existing caller was already
                         hardcoding before this parameter existed (see
                         WO#12 postmortem, Part 2 amendment 6), so nothing
                         already using post_with_retry changes behavior by
                         omitting it. Added so callers with different
                         latency needs (e.g. a multimodal/vision payload)
                         can override it without duplicating this whole
                         retry loop just to get a different timeout.

    Returns:
        Parsed JSON response body (resp.json()) on the first 2xx response.

    Raises:
        requests.HTTPError: on the first non-429/503 error response.
        RuntimeError: once `retries` attempts are exhausted on 429/503s.
    """
    for attempt in range(retries):
        resp = requests.post(url, json=payload, timeout=timeout)

        if resp.status_code == 429:
            wait = 30 * (attempt + 1)
            log.warning(
                "%s 429 on %s, waiting %ds (attempt %d/%d)",
                provider_name, resource_label, wait, attempt + 1, retries,
            )
            time.sleep(wait)
            continue

        if resp.status_code == 503:
            wait = 5 ** attempt
            log.warning(
                "%s 503 on %s, retrying in %ds (attempt %d/%d)",
                provider_name, resource_label, wait, attempt + 1, retries,
            )
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json()

    raise RuntimeError(
        f"{provider_name} {resource_label} unavailable after {retries} retries"
    )
