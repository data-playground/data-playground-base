# services/ai/__init__.py
#
# RECONSTRUCTED STUB — not provided as source material anywhere in this
# engagement (WO#11 through WO#16). Inferred purely from call-site usage.
# See the WO#16 postmortem's "Pre-Execution Gate" / "Known Caveats"
# sections: this stub has now accumulated exports from five consecutive
# work orders (WO#11, #13, #14, #15, #16) without ever being checked
# against the real repository file. VERIFY AGAINST THE REAL FILE before
# merging any of this.
import logging

from services.ai.providers.gemini import (
    MODEL_FLASH,
    MODEL_FLASH_LITE,
    MODEL_GEMMA,
    call_gemini_text,
    call_gemini_json,
    call_gemma_json,
    call_gemini_vision_json,
)
# ── Added by WO#14 (Groq provider + Ghostwriter migration) ─────────────────
from services.ai.providers.groq import (
    MODEL_LLAMA_70B,
    call_groq_text,
)
# ── Added by WO#15 (Cerebras provider migration) ────────────────────────────
from services.ai.providers.cerebras import (
    MODEL_QWEN3,
    MODEL_LLAMA33,
    call_cerebras_text,
)

log = logging.getLogger(__name__)

__all__ = [
    "MODEL_FLASH",
    "MODEL_FLASH_LITE",
    "MODEL_GEMMA",
    "call_gemini_text",
    "call_gemini_json",
    "call_gemma_json",
    "call_gemini_vision_json",
    "MODEL_LLAMA_70B",
    "call_groq_text",
    "MODEL_QWEN3",
    "MODEL_LLAMA33",
    "call_cerebras_text",
    "call_ai_text",
    "call_ai_json",
]


# ── WO#16: Generic, provider-agnostic dispatcher ────────────────────────────
# Additive only — does not replace or retrofit any of the direct
# provider-function calls used by job_agents.py, recipe_agents.py,
# weekly_agents.py, workout_plan_ai_generator.py, media_recommend.py, or
# blog_agents.py. All of those continue calling call_gemini_json() /
# call_groq_text() / call_cerebras_text() etc. directly — fully
# supported, not deprecated. Use call_ai_text()/call_ai_json() for new
# code going forward. See services/ai/README.md for fuller guidance.

def call_ai_text(
    provider: str,
    model: str,
    prompt: str,
    system: str | None = None,
    **kwargs,
) -> str:
    """
    Provider-agnostic text completion. Routes to the matching
    provider-specific function and normalizes the return to plain text.

    Args:
        provider: "gemini" | "groq" | "cerebras"
        model:    Provider-specific model ID string.
        prompt:   User-turn prompt text.
        system:   System instruction. call_gemini_text() and
                  call_groq_text() both require a system string
                  positionally (they don't support omitting it the way
                  call_gemini_json() does) — if None, an empty string is
                  passed through rather than raising.
        **kwargs: Passed through to the underlying provider function
                  (e.g. temperature, max_tokens, retries). Not validated
                  here — an argument the target function doesn't accept
                  surfaces as a normal TypeError from that function.

    Returns:
        Plain text content. For provider="cerebras" specifically, this
        discards the remaining_tokens value call_cerebras_text() returns
        as the second element of its (content, remaining_tokens) tuple.
        If you need that value (currently only
        blog_agents.py::agent_code_improver() does), call
        services.ai.call_cerebras_text() directly instead of this
        wrapper.

    Raises:
        ValueError: if `provider` isn't one of "gemini", "groq", "cerebras".
    """
    if provider == "gemini":
        return call_gemini_text(system or "", prompt, model=model, **kwargs)
    if provider == "groq":
        return call_groq_text(system or "", prompt, model=model, **kwargs)
    if provider == "cerebras":
        content, _remaining_tokens = call_cerebras_text(model, system or "", prompt, **kwargs)
        return content
    raise ValueError(
        f"Unknown AI provider: {provider!r}. Expected one of: "
        f"'gemini', 'groq', 'cerebras'."
    )


def call_ai_json(
    provider: str,
    model: str,
    prompt: str,
    schema: dict | None = None,
    system: str | None = None,
    **kwargs,
) -> str:
    """
    Provider-agnostic JSON-mode completion. Currently only "gemini" is
    supported — Groq and Cerebras have no JSON-mode caller anywhere in
    this codebase today, so there's nothing real to generalize their
    JSON behavior from yet.

    Routes to call_gemini_json() by default. If `model` equals
    MODEL_GEMMA, routes to call_gemma_json() instead, since Gemma's call
    shape has no systemInstruction support at all. In that case, if
    `system` was also provided, it's logged as a warning (dropped, not
    silently ignored) rather than raised.

    Raises:
        ValueError: if `provider` is not "gemini".
    """
    if provider != "gemini":
        raise ValueError(
            f"call_ai_json() only supports provider='gemini' currently, "
            f"got {provider!r}. Groq and Cerebras have no JSON-mode "
            f"caller in this codebase to generalize from yet — see this "
            f"function's docstring."
        )
    if model == MODEL_GEMMA:
        if system is not None:
            log.warning(
                "call_ai_json(provider='gemini', model=MODEL_GEMMA, ...) "
                "was given a system instruction, but call_gemma_json() "
                "has no systemInstruction support — it will be dropped. "
                "Prepend it into `prompt` yourself if needed."
            )
        return call_gemma_json(prompt, model=model, **kwargs)
    return call_gemini_json(prompt, schema=schema, system=system, model=model, **kwargs)
