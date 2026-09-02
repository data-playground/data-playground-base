# services/ai/__init__.py
#
# RECONSTRUCTED STUB — not provided in the source documents for this work
# order. VERIFY AGAINST THE REAL FILE before merging.
from services.ai.providers.gemini import (
    MODEL_FLASH,
    MODEL_FLASH_LITE,
    MODEL_GEMMA,
    call_gemini_text,
    call_gemini_json,
    call_gemma_json,
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

__all__ = [
    "MODEL_FLASH",
    "MODEL_FLASH_LITE",
    "MODEL_GEMMA",
    "call_gemini_text",
    "call_gemini_json",
    "call_gemma_json",
    "MODEL_LLAMA_70B",
    "call_groq_text",
    "MODEL_QWEN3",
    "MODEL_LLAMA33",
    "call_cerebras_text",
]
