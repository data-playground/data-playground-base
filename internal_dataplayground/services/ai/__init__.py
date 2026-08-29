# services/ai/__init__.py
#
# RECONSTRUCTED STUB — not provided in the source documents for this work
# order. Inferred from confirmed call sites:
#   job_agents.py:    from services.ai import call_gemini_json, MODEL_FLASH_LITE
#   recipe_agents.py: from services.ai import MODEL_FLASH, call_gemini_json, call_gemma_json
# VERIFY AGAINST THE REAL FILE before merging.
from services.ai.providers.gemini import (
    MODEL_FLASH,
    MODEL_FLASH_LITE,
    MODEL_GEMMA,
    call_gemini_text,
    call_gemini_json,
    call_gemma_json,
)

__all__ = [
    "MODEL_FLASH",
    "MODEL_FLASH_LITE",
    "MODEL_GEMMA",
    "call_gemini_text",
    "call_gemini_json",
    "call_gemma_json",
]
