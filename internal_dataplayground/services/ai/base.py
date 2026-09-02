# services/ai/base.py
#
# RECONSTRUCTED STUB — not provided in the source documents for this work
# order. VERIFY AGAINST THE REAL FILE before merging.
import logging
import time

import requests

log = logging.getLogger(__name__)

_DEFAULT_BACKOFF = [1, 5, 25]


def post_with_retry(
    url: str,
    payload: dict,
    retries: int = 3,
    provider_name: str = "",
    resource_label: str = "",
    timeout: float = 90.0,
) -> dict:
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
