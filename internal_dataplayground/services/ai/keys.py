# services/ai/keys.py
"""
Centralized API key/secret lookup for the AI Service Layer (GOVERNANCE.md §2.3).

Wraps gcp_secrets.get_key() where that's compatible with a given provider's
existing lookup convention. Does NOT replace gcp_secrets.py — this module
is a thin, provider-name-keyed convenience wrapper that services/ai/
provider modules import instead of each reaching for os.environ.get(...)
directly, the way every current agent module (gemini_client.py,
blog_agents.py, recipe_agents.py, weekly_agents.py) does independently
today.

GEMINI KEY LOOKUP — NOW ROUTED THROUGH gcp_secrets.get_key():
  Every existing caller resolves the Gemini key via a private
  `_gemini_key()` (or equivalent) that does exactly:

      os.environ.get("GEMINI_API")

  gcp_secrets.py's source (now available — see Work Order #11 addendum)
  shows get_key(secret_name) does:

      env_key = secret_name.upper().replace("-", "_")
      env_val = os.environ.get(env_key)
      if env_val: return env_val
      # else: falls back to a live GCP Secret Manager call for a secret
      #       literally named `secret_name`.

  For get_key("GEMINI_API"): env_key normalizes to "GEMINI_API" (already
  uppercase, no hyphens) — so the env-var branch is byte-for-byte
  os.environ.get("GEMINI_API"), identical to the pre-migration lookup.
  Since docker-compose.yml always injects GEMINI_API into both the `web`
  and `airflow-*` containers, that branch is what actually executes in
  every real deployment of this app — so the returned value is unchanged
  in the case that matters. google-cloud-secret-manager is already a hard
  dependency (database.py already imports it), so this adds no new risk.

  KNOWN DIVERGENCE (edge case only, never hit today): if GEMINI_API were
  ever unset, the old code silently returned None; get_key() instead
  attempts a live Secret Manager call for a secret named "GEMINI_API" —
  which either succeeds with a real value or raises, rather than failing
  silently later at the HTTP-call layer with key=None in the URL. This is
  arguably a strict improvement (fail loud vs. fail with a broken request)
  but it IS a behavior change in that one edge case, flagged here rather
  than absorbed silently.
"""

from gcp_secrets import get_key as _gcp_get_key

# Provider name -> secret name, mirroring the env var names already used
# in docker-compose.yml and every existing agent module. Passed straight
# into gcp_secrets.get_key(), which uppercases/normalizes internally.
_PROVIDER_SECRET_NAMES = {
    "gemini":   "GEMINI_API",
    "groq":     "GROQ_API",
    "cerebras": "CEREBRAS_API",
}


def get_provider_key(provider: str) -> str:
    """
    Returns the API key/secret for a given provider name
    ("gemini", "groq", "cerebras", ...) via gcp_secrets.get_key().

    For "gemini" specifically, this returns exactly what gemini_client.py's
    pre-migration `_gemini_key()` returned in every real deployment of
    this app (env var always present via docker-compose.yml) — see module
    docstring for the trace and the one known edge-case divergence.

    "groq" and "cerebras" are included for forward compatibility with the
    providers those will need once WO#12+ builds them — nothing calls
    get_provider_key("groq"/"cerebras") yet.

    Args:
        provider: Provider name. Must be one of _PROVIDER_SECRET_NAMES.

    Returns:
        The key/secret value.

    Raises:
        ValueError: if `provider` isn't a recognized provider name.
    """
    if provider not in _PROVIDER_SECRET_NAMES:
        raise ValueError(
            f"Unknown AI provider '{provider}'. "
            f"Known providers: {sorted(_PROVIDER_SECRET_NAMES)}"
        )
    return _gcp_get_key(_PROVIDER_SECRET_NAMES[provider])
