# services/ai/keys.py
#
# RECONSTRUCTED STUB — not provided in the source documents for this work
# order. Inferred purely from services/ai/providers/gemini.py's call to
# get_provider_key('gemini'). VERIFY AGAINST THE REAL FILE before merging.
import os

_ENV_VAR_BY_PROVIDER = {
    "gemini": "GEMINI_API",
    "groq": "GROQ_API",
    "cerebras": "CEREBRAS_API",
}


def get_provider_key(provider: str) -> str:
    """
    Looks up the API key for a given provider name from the environment.

    Raises RuntimeError if the env var is unset/empty rather than
    silently returning None — matches the pattern
    airflow/agents/media_agents.py's own _tmdb_key() already uses (fixed
    here after that inconsistency was noticed post-WO#13; a silently
    missing key would otherwise surface downstream as a confusing
    provider-side auth error instead of a clear "key not set" message).
    """
    env_var = _ENV_VAR_BY_PROVIDER.get(provider)
    if env_var is None:
        raise ValueError(f"Unknown AI provider: {provider!r}")
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"{env_var} is not set in the environment "
            f"(required for AI provider {provider!r})"
        )
    return key
