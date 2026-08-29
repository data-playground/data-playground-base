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
    """Looks up the API key for a given provider name from the environment."""
    env_var = _ENV_VAR_BY_PROVIDER.get(provider)
    if env_var is None:
        raise ValueError(f"Unknown AI provider: {provider!r}")
    return os.environ.get(env_var)
