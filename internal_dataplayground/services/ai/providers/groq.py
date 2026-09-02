# services/ai/providers/groq.py
"""
Groq provider implementation for the AI Service Layer (GOVERNANCE.md §2.3).

Moved from airflow/agents/blog_agents.py's `_groq_llama()` — used solely
by the Ghostwriter agent for prose generation. Behavior is unchanged:
same endpoint, same OpenAI-compatible chat/completions payload shape
(system + user messages, not a Gemini-style systemInstruction field),
same temperature/max_tokens defaults, no retry/backoff logic (the
original function had none, so none is added here — see WO#14 HARD
BOUNDARIES).

The private `_groq_key()` helper that lived alongside `_groq_llama()` in
blog_agents.py is replaced by `services.ai.keys.get_provider_key("groq")`.
"""
import requests

from services.ai.keys import get_provider_key

MODEL_LLAMA_70B = "llama-3.3-70b-versatile"

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def call_groq_text(
    system: str,
    prompt: str,
    model: str = MODEL_LLAMA_70B,
    temperature: float = 0.7,
    max_tokens: int = 8192,
) -> str:
    """
    Calls Groq's OpenAI-compatible chat/completions endpoint.
    Used for prose generation (currently: the Ghostwriter agent only).

    No retry/backoff logic — matches the original `_groq_llama()`, which
    made a single `requests.post()` call and relied on `raise_for_status()`
    alone. Do not add retry logic here without a dedicated work order;
    that would be a behavior change, not a relocation.
    """
    headers = {
        "Authorization": f"Bearer {get_provider_key('groq')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(_GROQ_URL, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
