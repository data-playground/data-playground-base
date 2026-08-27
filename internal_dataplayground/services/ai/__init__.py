# services/ai/__init__.py
"""
AI Service Layer — shared, provider-backed AI call helpers (GOVERNANCE.md §2.3).

Public contract established in WO#11: re-export the Gemini-specific
call_gemini_text / call_gemini_json, plus both model constants, at the top
level:

    from services.ai import call_gemini_json, MODEL_FLASH_LITE

MODEL_FLASH_LITE (and MODEL_FLASH) are re-exported here too, not just left
in services.ai.providers.gemini, so that a caller's migration can be a
single import line:

    from services.ai import call_gemini_json, MODEL_FLASH_LITE

rather than importing from two different module paths for what is, from
the caller's point of view, one logical change.

WO#12 extended this same top-level contract with the Gemma variant needed
by recipe_agents.py's ingredient normalizer — Gemma is served through the
same Gemini API surface (same base URL, same key), but the request shape
differs (no systemInstruction support, no responseSchema enforcement), so
it gets its own function rather than an optional flag on call_gemini_json:

    from services.ai import call_gemma_json, MODEL_GEMMA

All three are just thin wrappers around services.ai.providers.gemini —
this module's job is purely to be the one place callers import from,
not to add logic of its own.

The eventual target interface described in GOVERNANCE.md §2.3 is a
provider-agnostic `call_ai_text()` / `call_ai_json()` pair that dispatches
across providers based on config rather than the caller importing a
specific provider's function. That wrapper is intentionally NOT added
here. Designing it correctly needs at least two real providers to
generalize from — Groq (blog_agents.py's _groq_llama) and Cerebras
(blog_agents.py's _cerebras) have meaningfully different call shapes
(different retry semantics, Cerebras returns a tuple), and guessing at
the provider-selection/config shape from Gemini alone risks getting it
wrong. That wrapper is deferred to the work order that adds the second
provider. See docs/ai-service-layer-postmortem.md for the running list of
what's migrated, what's deliberately deferred, and what full-migration
cleanup still needs to happen.
"""

from services.ai.providers.gemini import (
    MODEL_FLASH,
    MODEL_FLASH_LITE,
    MODEL_GEMMA,
    call_gemini_json,
    call_gemini_text,
    call_gemma_json,
)

__all__ = [
    "MODEL_FLASH",
    "MODEL_FLASH_LITE",
    "MODEL_GEMMA",
    "call_gemini_json",
    "call_gemini_text",
    "call_gemma_json",
]
