# Work Order #12 — AI Service Layer: Migrate `recipe_agents.py`

*Second caller migration (see WO#11 for the foundation this builds on).
`recipe_agents.py` uses three call shapes: `_gemini_flash` and
`_gemini_flash_json` (already covered by `services.ai.call_gemini_text` /
`call_gemini_json` from WO#11 — straightforward import swap), plus `_gemma`
(a Gemma-model variant with no system-instruction support and no schema
enforcement — new to the service layer, added here). One function in this
file, `agent_extract_recipe_from_image`, uses a custom vision payload
(base64 image + inline data) that doesn't fit any existing service-layer
function — it is explicitly OUT OF SCOPE for this work order and stays on
its raw implementation until a future work order adds vision support to
the service layer.*

---

## ROLE
You are a senior refactoring engineer consolidating duplicated
infrastructure code. Your job is to route existing callers through the
shared `services/ai/` layer built in WO#11 — not to redesign prompts,
change retry behavior, or "improve" anything about how these agents work.
Flag opportunities in NOTES; do not act on them.

## HARD BOUNDARIES
- Only read/edit files explicitly listed in SCOPE below.
- **Do not touch `blog_agents.py`, `weekly_agents.py`,
  `workout_plans.py`'s `_call_gemini_for_plan`, `media_recommend.py`'s
  `_gemini_explain`, or `finance_upload.py`'s Gemini SDK call.** Same
  exclusion list as WO#11, still in force — those remain separate future
  work orders.
- **`agent_extract_recipe_from_image()` is explicitly OUT OF SCOPE.** Its
  raw `requests.post(...)` call with an `inlineData` image part and vision
  prompt does not fit `call_gemini_text`/`call_gemini_json`'s signatures.
  Do not attempt to force it through the existing service-layer functions,
  and do not build vision support into `services/ai/` as part of this work
  order — that's real new-capability design, not a mechanical migration,
  and deserves its own scoped work order. Leave this function's
  implementation completely untouched, including its direct use of
  `_gemini_key()` (which stays in `recipe_agents.py` for this one function
  only — see Step 4).
- No behavior changes to prompts, schemas, model IDs, or retry timing.
  Same values, same logic, just relocated to the shared layer.
- The `_safe_json()` markdown-fence-stripping helper in `recipe_agents.py`
  is domain-specific JSON cleanup, not a provider-call concern — it stays
  in `recipe_agents.py` unchanged. Do not move it into `services/ai/`.

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

**Existing service-layer files to extend (built in WO#11):**
- `services/ai/providers/gemini.py`
- `services/ai/__init__.py`

**File to edit (the caller being migrated):**
- `airflow/agents/recipe_agents.py`

**Not in scope, referenced only to confirm no breakage:**
- `routers/recipe_extract.py`, `routers/recipe_discovery.py`,
  `services/recipe_service.py` (all call `recipe_agents.py`'s public
  `agent_*` functions, not its private `_gemini_flash`/`_gemma` helpers
  directly — should need zero changes; confirm rather than assume)
- **Note on router paths:** depending on whether Work Order #7 (`recipes`
  domain migration) has been run yet, `recipe_extract.py` and
  `recipe_discovery.py` may be at their original `routers/` location or
  at `domains/recipes/routers/`. Locate them wherever they currently are
  — this work order does not depend on WO#7 having run first, since
  `recipe_agents.py` itself was explicitly excluded from that migration
  (it stays under `airflow/agents/` regardless of domain restructuring
  status) and its callers only need confirming, not editing.

---

## STEPS

1. **Add a Gemma-variant function to `services/ai/providers/gemini.py`.**
   Extract `recipe_agents.py`'s `_gemma()` logic into a new function:
   ```python
   MODEL_GEMMA = "gemma-4-31b-it"

   def call_gemma_json(prompt: str, model: str = MODEL_GEMMA) -> str:
       """
       Calls a Gemma model via the Gemini API endpoint. Unlike
       call_gemini_json(), Gemma models don't support systemInstruction —
       callers must prepend any system context directly into `prompt`
       themselves, same as recipe_agents.py's original _gemma() required.
       No responseSchema enforcement — only responseMimeType: application/json.
       """
   ```
   Build it on `services.ai.base`'s shared retry function, same as the
   existing `call_gemini_text`/`call_gemini_json`. Use
   `services.ai.keys.get_provider_key("gemini")` for the API key (Gemma is
   served through the same Gemini API surface, same key).

2. **Export `call_gemma_json` and `MODEL_GEMMA`** from
   `services/ai/__init__.py` alongside the existing `call_gemini_text`/
   `call_gemini_json` re-exports.

3. **Update `recipe_agents.py`'s three text/JSON callers:**
   - Replace `_gemini_flash(system, prompt)` calls with `from services.ai
     import call_gemini_text` and update call sites accordingly (same
     argument order).
   - Replace `_gemini_flash_json(system, prompt, schema)` calls with
     `call_gemini_json` similarly.
   - Replace `_gemma(prompt)` calls (inside `agent_normalize_ingredients`)
     with the new `call_gemma_json(prompt)`.
   - Delete the now-unused private `_gemini_flash()`, `_gemini_flash_json()`,
     and `_gemma()` function definitions from `recipe_agents.py` once their
     call sites are updated.

4. **Leave `_gemini_key()` in `recipe_agents.py`, used only by
   `agent_extract_recipe_from_image()`.** Since that function is out of
   scope (see HARD BOUNDARIES), its dependency on the private
   `_gemini_key()` helper must remain — do not delete `_gemini_key()`
   entirely, only stop using it in the three functions migrated in Step 3.
   Add a one-line comment above the remaining `_gemini_key()` definition
   noting it's kept specifically for the not-yet-migrated vision function.

---

## ACCEPTANCE CRITERIA

- [ ] `agent_extract_recipe()` (uses `call_gemini_json`) produces an
  identical request to its pre-migration behavior for the same inputs —
  verify via mocked HTTP layer, mark ⚠️ if live verification isn't
  available
- [ ] `agent_normalize_ingredients()` (uses the new `call_gemma_json`)
  produces an identical request — same endpoint, same model ID, no
  `systemInstruction` key in the payload, `responseMimeType:
  "application/json"` present, no `responseSchema` key — matching the
  original `_gemma()` exactly
- [ ] `agent_discover_recipes_pantry()` and `agent_discover_recipes_open()`
  (both use `call_gemini_json`) still build their prompts/schemas
  identically and route through the new import correctly
- [ ] `agent_extract_recipe_from_image()` is **completely unchanged** —
  `git diff` on this specific function should show zero lines changed;
  confirm explicitly in your report
- [ ] `recipe_agents.py` no longer defines `_gemini_flash`,
  `_gemini_flash_json`, or `_gemma` — confirm via `grep` within the file
- [ ] `recipe_agents.py` still defines `_gemini_key()`, used only by the
  untouched vision function
- [ ] `routers/recipe_extract.py`, `routers/recipe_discovery.py` (or their
  `domains/recipes/routers/` equivalents, whichever currently exists), and
  `services/recipe_service.py` require **zero** changes — confirm by
  reviewing their imports of `recipe_agents` (they call public `agent_*`
  functions only)
- [ ] `grep -r "_gemini_flash\|_gemini_flash_json" airflow/agents/recipe_agents.py`
  returns zero hits (confirming full removal of the now-dead private
  functions, not just their replacement at call sites)

---

## For the next work order (not part of this one)

**Work Order #13** should tackle `weekly_agents.py` (its own
`_gemini_flash_json`, another straightforward swap onto
`call_gemini_json`, similar shape to `recipe_agents.py`'s migration here)
and, if convenient to batch together, the inline calls in
`workout_plans.py` (`_call_gemini_for_plan`) and `media_recommend.py`
(`_gemini_explain`) — both are also plain JSON-mode Gemini calls with no
new capability needed, unlike the Gemma or vision cases. That would leave
only `_groq_llama` (a genuinely different provider — needs
`services/ai/providers/groq.py`, not yet built), `_cerebras` (different
provider with more complex retry/backoff — needs
`services/ai/providers/cerebras.py`), `finance_upload.py`'s SDK-based call
(a different calling convention entirely, worth a design discussion before
migrating rather than a mechanical port), and the vision-support gap
flagged in this work order's HARD BOUNDARIES. At that point, with three
Gemini-shaped callers fully migrated, it's also the right time to finally
design the provider-agnostic `call_ai_text()`/`call_ai_json()` wrapper
mentioned in GOVERNANCE.md §2.3 and WO#11's closing note — there will be
enough real call-site shapes by then to generalize from with confidence.
