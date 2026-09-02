# services/ai/providers/cerebras.py
"""
Cerebras provider implementation for the AI Service Layer (GOVERNANCE.md §2.3).

Moved from airflow/agents/blog_agents.py's `_cerebras()` — used by four
agents (Code Narrator, Refiner, Code Commenter, Code Improver). This is a
verbatim relocation, not a rewrite: the retry/backoff schedule
(_CEREBRAS_BACKOFF = [75, 150, 300, 600]s), the dual handling of 429s
(both the SDK's raw-response 429 status code AND the RateLimitError
exception path) and 503s (both the raw-response 503 status code AND the
APIStatusError exception path), the Retry-After header override on 429,
the token-remaining header parsing (with its (ValueError, TypeError)
fallback to 0), and the final RuntimeError-with-last_exc-detail on
exhausted retries are all preserved exactly as they were in
blog_agents.py. See the WO#15 postmortem for the mocked-SDK verification
that confirmed this.

The private `_cerebras_key()` helper that lived alongside `_cerebras()` in
blog_agents.py is replaced by `services.ai.keys.get_provider_key("cerebras")`.

Return shape is preserved exactly: (content: str, remaining_tokens: int).
`agent_code_improver()` in blog_agents.py depends on this tuple shape —
`life_os_code_improve.py`'s DAG task uses the remaining-token count to
decide whether to sleep between files. Do not change this return shape
without also updating that DAG (out of scope for this provider module).
"""
import logging

from services.ai.keys import get_provider_key

log = logging.getLogger(__name__)

MODEL_QWEN3 = "qwen-3-235b-a22b-instruct-2507"
MODEL_LLAMA33 = "llama-3.3-70b"

# Default backoff schedule for 429 responses (seconds).
# Cerebras resets its RPM window every 60 seconds, so the max wait
# is capped there. If Retry-After header is present it overrides this.
_CEREBRAS_BACKOFF = [75, 150, 300, 600]


def call_cerebras_text(
    model: str,
    system: str,
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> tuple[str, int]:
    """
    Calls a Cerebras-hosted model with production-tuned retry/backoff
    for rate limiting. Returns (content, remaining_tokens_this_minute).
    Callers that don't need the remaining-token count can discard it:
    content, _ = call_cerebras_text(...)
    """
    log.info("call_cerebras_text() — retry loop active, backoff=%s", _CEREBRAS_BACKOFF)

    import time

    from cerebras.cloud.sdk import APIStatusError, Cerebras, RateLimitError

    client = Cerebras(api_key=get_provider_key("cerebras"), max_retries=0).with_raw_response
    last_exc = None

    for attempt, wait in enumerate(_CEREBRAS_BACKOFF):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            remaining_day = resp.headers.get("x-ratelimit-remaining-tokens-day", "?")
            remaining_min = resp.headers.get("x-ratelimit-remaining-tokens-minute", "?")
            reset_min     = resp.headers.get("x-ratelimit-reset-tokens-minute", "?")

            if resp.status_code == 200:
                log.info(
                    "Cerebras %s OK — tokens remaining: %s/day, %s/min (reset in %ss)",
                    model, remaining_day, remaining_min, reset_min,
                )
                content = resp.json()["choices"][0]["message"]["content"]

                try:
                    remaining_min_int = int(remaining_min)
                except (ValueError, TypeError):
                    remaining_min_int = 0
                return content, remaining_min_int

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                actual_wait = float(retry_after) if retry_after else wait
                log.warning(
                    "Cerebras 429 on attempt %d/%d. Waiting %.1fs.",
                    attempt + 1, len(_CEREBRAS_BACKOFF), actual_wait,
                )
                last_exc = RuntimeError(f"Cerebras 429 on attempt {attempt + 1}")
                time.sleep(actual_wait)
                continue

            if resp.status_code == 503:
                log.warning("Cerebras 503 on attempt %d/%d. Waiting %ds.",
                            attempt + 1, len(_CEREBRAS_BACKOFF), wait)
                last_exc = RuntimeError(f"Cerebras 503 on attempt {attempt + 1}")
                time.sleep(wait)
                continue

            resp.raise_for_status()

        except RateLimitError as exc:
            response = getattr(exc, "response", None)
            retry_after = 60  # safe default
            if response is not None:
                try:
                    retry_after = int(getattr(response, "headers", {}).get("retry-after", 60))
                except (ValueError, TypeError):
                    pass
            log.warning(
                "Cerebras 429 RateLimitError on attempt %d/%d. Waiting %ds.",
                attempt + 1, len(_CEREBRAS_BACKOFF), retry_after,
            )
            last_exc = exc
            time.sleep(retry_after)
            continue

        except APIStatusError as exc:
            if exc.status_code == 503:
                log.warning("Cerebras APIStatusError 503 on attempt %d/%d. Waiting %ds.",
                            attempt + 1, len(_CEREBRAS_BACKOFF), wait)
                last_exc = exc
                time.sleep(wait)
                continue
            log.error("Cerebras non-retriable APIStatusError: %s", exc)
            raise

        except Exception as exc:
            log.error("Cerebras unexpected error: %s", exc)
            raise

    raise RuntimeError(
        f"Cerebras {model} unavailable after {len(_CEREBRAS_BACKOFF)} retries. "
        f"Last error: {last_exc}"
    )
