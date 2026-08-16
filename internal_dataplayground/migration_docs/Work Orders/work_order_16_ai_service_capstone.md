# Work Order #16 — AI Service Layer: Generic Dispatcher, Vision Support, and Closing Decisions

*Capstone work order for the AI Service Layer effort (GOVERNANCE.md §2.3,
WO#11–15). With Gemini, Groq, and Cerebras all now in
`services/ai/providers/`, this work order (1) adds the provider-agnostic
`call_ai_text()`/`call_ai_json()` wrapper — as a convenience for *future*
callers, not a forced migration of the five already-working callers from
WO#11–15, (2) adds vision support to close the gap flagged in WO#12, and
(3) makes an explicit, documented decision about `finance_upload.py`'s
SDK-based call rather than leaving it as an open question indefinitely.
After this work order, GOVERNANCE.md §2.3's "target state" is fully
realized except for `blog_agents.py`'s dead-code cleanup, which stays a
separate, smaller, later task per WO#15's closing note.*

---

## ROLE
You are a senior refactoring engineer closing out a multi-phase
consolidation effort. Your job is to add the generalized interface now
that three real provider shapes exist to inform its design, close one
genuine capability gap (vision), and make one deliberate, documented scope
decision (finance_upload.py) rather than let it drift unresolved. This is
still not a license to touch any of the five callers already migrated in
WO#11–15 — they keep working exactly as they do today, calling their
provider-specific functions directly. The generic dispatcher is additive,
not a replacement for what already exists and already works.

## HARD BOUNDARIES
- Only read/edit files explicitly listed in SCOPE below.
- **Do not migrate `job_agents.py`, `recipe_agents.py`,
  `weekly_agents.py`, `workout_plans.py`, `media_recommend.py`, or
  `blog_agents.py`'s Ghostwriter/Narrator/Refiner/Commenter/Improver
  functions to use the new generic dispatcher.** They already call
  `services.ai.call_gemini_json` / `call_groq_text` / `call_cerebras_text`
  directly, which is a completely valid, supported usage pattern — the
  generic dispatcher is for *new* code going forward, not a mandatory
  migration target for existing working code. Retrofitting them would be
  scope creep with real regression risk for zero functional benefit.
- **The generic `call_ai_text()` wrapper normalizes away Cerebras's
  `remaining_tokens` return value** (returns only the text content). Any
  caller that needs `remaining_tokens` (currently only
  `agent_code_improver()`) must continue calling
  `services.ai.call_cerebras_text()` directly, not the generic wrapper.
  Document this limitation clearly in the dispatcher's docstring — do not
  try to design around it (e.g. with an optional "return metadata" flag)
  in this pass; that's speculative design for a need that doesn't
  currently exist beyond the one already-excluded caller.
- **`finance_upload.py`'s `google-genai` SDK-based Gemma call is
  explicitly NOT migrated in this work order** — see Step 4 for the
  documented decision to leave it as an intentional exception. Do not
  attempt to convert it to the raw-REST pattern.
- Vision support (Step 3) is scoped narrowly to matching
  `agent_extract_recipe_from_image()`'s exact current request shape — do
  not build a general-purpose "attach any file type" abstraction.

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
If, while verifying an acceptance criterion, you discover the app behaves
incorrectly in a way that is unrelated to this migration:
1. Do NOT fix it — it is out of scope by default even if the fix looks trivial.
2. Reproduce the same failure against the pre-migration baseline to confirm
   it is not a regression you introduced.
3. Report it under "Notes" with enough detail to file a standalone ticket.
4. Mark the related acceptance criterion ⚠️ (not ❌) if the criterion's
   *intent* is otherwise satisfied but blocked by this pre-existing issue.

## WORKING METHOD
Execute steps in order. Verify incrementally, not only at the end. If an
acceptance criterion needs a live API call that isn't available here,
verify request construction via a mocked HTTP layer instead, state the
substitution explicitly, and mark ⚠️.

## OUTPUT FORMAT
1. **Files created**
2. **Files moved**
3. **Files edited** (path — description; flag/explain anything beyond the
   literal instructions)
4. **Acceptance criteria results** (✅/❌/⚠️ + one-line reason for non-✅)
5. **Notes**

## ROLLBACK
`git checkout` on every file listed in sections 1–3 of the output above.

---

## SCOPE

**Files to edit/extend:**
- `services/ai/__init__.py` (add the generic dispatcher)
- `services/ai/providers/gemini.py` (add vision support)

**New file to create:**
- `services/ai/README.md` (the model-routing rationale, moved from
  `blog_agents.py`'s header docstring per GOVERNANCE.md §2.3, plus the
  generic-vs-specific usage guidance and the `finance_upload.py` decision)

**File to edit (vision migration, one function only):**
- `airflow/agents/recipe_agents.py` — specifically
  `agent_extract_recipe_from_image()` only. No other function in this file
  changes (it was already fully migrated except this one function in
  WO#12).

**File to edit (documentation only, no code change):**
- `routers/finance_upload.py` (or `domains/finance/routers/finance_upload.py`
  if WO#5 has already run — locate wherever it currently is) — add a code
  comment documenting the Step 4 decision, no functional change

**Not in scope, referenced only to confirm no breakage:**
- `job_agents.py`, `recipe_agents.py`'s other functions, `weekly_agents.py`,
  `workout_plans.py`, `media_recommend.py`, `blog_agents.py`'s other
  functions — all confirmed unchanged, per HARD BOUNDARIES

---

## STEPS

1. **Design and add the generic dispatcher to `services/ai/__init__.py`.**
   Based on the three real shapes now in `services/ai/providers/`:
   ```python
   def call_ai_text(
       provider: str,       # "gemini" | "groq" | "cerebras"
       model: str,
       prompt: str,
       system: str | None = None,
       **kwargs,             # passed through to the provider function
                              # (e.g. temperature, max_tokens, retries)
   ) -> str:
       """
       Provider-agnostic text completion. Routes to the matching
       provider-specific function and normalizes the return to plain text.

       For Cerebras specifically: this discards the remaining_tokens
       value returned by the underlying call. If you need that value
       (currently only agent_code_improver() does), call
       services.ai.call_cerebras_text() directly instead of this wrapper.

       This function does not retrofit or replace any of the existing
       direct provider-function calls used by job_agents.py,
       recipe_agents.py, weekly_agents.py, workout_plans.py,
       media_recommend.py, or blog_agents.py — all of those continue
       calling their provider-specific functions directly, which remains
       fully supported. Use call_ai_text()/call_ai_json() for new code
       going forward.
       """
   ```
   Implement by dispatching on `provider` to `call_gemini_text`,
   `call_groq_text`, or `call_cerebras_text` (discarding the tuple's
   second element for the Cerebras case). Raise a clear `ValueError` for
   an unrecognized provider name rather than failing silently.

   Add the equivalent `call_ai_json()`:
   ```python
   def call_ai_json(
       provider: str,        # "gemini" only, for now — see docstring
       model: str,
       prompt: str,
       schema: dict | None = None,
       system: str | None = None,
       **kwargs,
   ) -> str:
       """
       Provider-agnostic JSON-mode completion. Currently only "gemini" is
       supported (routes to call_gemini_json, or call_gemma_json if the
       model name matches a known Gemma model ID) — Groq and Cerebras
       have no JSON-mode caller anywhere in the codebase today, so there
       is nothing real to generalize their JSON behavior from yet. Add
       support for another provider here only once a real caller needs
       it, following the same "generalize from real usage" principle
       used throughout this migration series.
       """
   ```

2. **Confirm backward compatibility.** Run through each of the five
   already-migrated callers' existing test/verification paths from
   WO#11–15 (or re-verify equivalent behavior) to confirm none of them
   were accidentally touched by this addition. This should be a quick
   confirmation, not a re-run of every prior work order's full acceptance
   criteria — just confirm `services/ai/__init__.py`'s existing exports
   (`call_gemini_text`, `call_gemini_json`, `call_gemma_json`,
   `call_groq_text`, `call_cerebras_text`, and their associated model
   constants) are still present and unchanged, since the new dispatcher
   functions are additions, not replacements.

3. **Add vision support to `services/ai/providers/gemini.py`.**
   ```python
   def call_gemini_vision_json(
       system: str,
       image_base64: str,
       mime_type: str,
       prompt: str,
       schema: dict,
       model: str = MODEL_FLASH,
   ) -> str:
       """
       Calls Gemini with an image (inlineData) plus a text prompt,
       enforcing JSON schema output. Used for image-based extraction
       (currently: recipe photo extraction).
       """
   ```
   Match `agent_extract_recipe_from_image()`'s current raw payload
   structure exactly: `systemInstruction`, `contents` with both an
   `inlineData` part (`mimeType`, `data`) and a `text` part, and
   `generationConfig` with `responseMimeType`/`responseSchema`. Build it
   on the shared retry logic from `services/ai/base.py`, same as every
   other Gemini function.

   Update `recipe_agents.py::agent_extract_recipe_from_image()` to call
   `from services.ai import call_gemini_vision_json` (or import from
   `services.ai.providers.gemini` directly — be consistent with how the
   other functions in this file already import, which per WO#12 is `from
   services.ai import ...`) instead of its own inline
   `requests.post(...)`. Remove the now-fully-unused `_gemini_key()`
   helper from `recipe_agents.py` **only if** this was genuinely its last
   caller — confirm via grep first (per WO#12's Step 4 note, it was kept
   specifically for this one function).

4. **Document the `finance_upload.py` decision.** Add a comment directly
   above its Gemini/Gemma call in `_categorise_batch()`:
   ```python
   # NOTE: This uses the google-genai SDK directly rather than
   # services/ai/ (see services/ai/README.md "SDK Exceptions" section).
   # Decision: left as-is — the SDK handles response parsing and model
   # selection differently enough from the raw-REST pattern used
   # elsewhere that converting it doesn't reduce duplication, it just
   # changes which duplication exists. Revisit only if a second
   # SDK-based caller appears, at which point the SDK pattern itself
   # may be worth its own services/ai/providers/ entry.
   ```
   No functional change to this function — comment only.

5. **Create `services/ai/README.md`.** Contents:
   - The full model-routing rationale table and prose currently living in
     `blog_agents.py`'s module docstring (Agent / Provider / Model table,
     and the "ROUTING RATIONALE" section explaining why each provider was
     chosen for each task category) — copied verbatim, not rewritten.
     **Do not delete it from `blog_agents.py`** — leave the original in
     place too (this is documentation duplication, not code duplication,
     and removing institutional knowledge from its original, well-known
     location is riskier than a little redundancy; a future cleanup work
     order can consolidate to one location once people are used to
     checking the new one).
   - A short "When to use `call_ai_text`/`call_ai_json` vs. a
     provider-specific function" section, capturing the guidance from
     Step 1's docstrings in one discoverable place.
   - The `finance_upload.py` SDK-exception decision from Step 4, written
     out in full (the code comment can reference this file for the full
     reasoning).
   - A short "Provider coverage" list: Gemini (text, JSON w/ schema, JSON
     w/o schema, Gemma variant, vision), Groq (text), Cerebras (text with
     retry/backoff + token-remaining metadata) — so a future contributor
     can see at a glance what's already supported before adding something
     new.

---

## ACCEPTANCE CRITERIA

- [ ] `call_ai_text(provider="gemini", ...)`, `call_ai_text(provider="groq",
  ...)`, and `call_ai_text(provider="cerebras", ...)` each correctly
  dispatch to their respective provider function and return a plain string
  (Cerebras case confirmed to discard `remaining_tokens`, not raise or
  return a tuple)
- [ ] `call_ai_text(provider="nonexistent", ...)` raises `ValueError`
  rather than failing with an unrelated error or silently doing nothing
- [ ] `call_ai_json(provider="gemini", ...)` correctly dispatches to
  `call_gemini_json`
- [ ] `call_gemini_vision_json` produces an identical request (system
  instruction, inline image data, text prompt, schema) to what
  `agent_extract_recipe_from_image()`'s original raw implementation
  produced — verify via mocked HTTP layer, mark ⚠️ if live verification
  isn't available
- [ ] `agent_extract_recipe_from_image()` still returns the same
  validated/sanitized dict structure as before (meal_type/difficulty enum
  validation, `raw_ingredient_lines` list-type enforcement — this
  post-processing logic in the function body is unchanged, only the HTTP
  call itself was replaced)
- [ ] **Confirm zero changes** to `job_agents.py`, `weekly_agents.py`,
  `workout_plans.py`, `media_recommend.py`, and every `blog_agents.py`
  function except none (blog_agents.py isn't touched by this work order
  at all) — `git diff` on each should be empty or absent from the changed
  file list entirely
- [ ] `recipe_agents.py`'s other four migrated functions (from WO#12) are
  unchanged — only `agent_extract_recipe_from_image()`'s implementation
  changed
- [ ] `finance_upload.py`'s `_categorise_batch()` behavior is byte-for-byte
  unchanged (comment-only edit) — confirm via `git diff` showing only
  comment lines added
- [ ] `services/ai/README.md` exists and covers all four required
  sections (routing rationale, generic-vs-specific guidance, SDK
  exception decision, provider coverage list)
- [ ] `blog_agents.py`'s module docstring still contains the original
  routing rationale — confirm it was NOT deleted, only duplicated into the
  new README

---

## Series Wrap-Up (applies after this work order, not part of it)

This closes the planned AI Service Layer work (WO#11–16). What remains,
tracked but intentionally not scoped into any of these sixteen work
orders:
1. **`blog_agents.py` dead-code cleanup** — the commented-out prior
   `_cerebras()` implementation (flagged in WO#15) and any now-redundant
   section headers. Small, low-risk, do whenever convenient.
2. **Retrofitting existing callers onto the generic dispatcher** —
   explicitly not done in this series (see HARD BOUNDARIES throughout
   WO#11–16) and should stay that way unless a concrete reason emerges
   (e.g. a new cross-provider feature that genuinely needs the
   abstraction). Do not treat "the generic wrapper exists now" as
   sufficient reason on its own.
3. **Frontend consolidation (GOVERNANCE.md §2.3.1 / §3.2)** — toast
   notifications, sidebar/theme JS duplication, inline-style cleanup.
   Never scoped into any work order in this series; still fully
   outstanding and worth its own dedicated pass whenever domain migration
   and AI consolidation work has settled.
4. Everything else already tracked in GOVERNANCE.md §5 (adaptive
   dashboard, digest email, in-Docker coding environments, new domains) —
   unchanged status, still deliberately parked.
