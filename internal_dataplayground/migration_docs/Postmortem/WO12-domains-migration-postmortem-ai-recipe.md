# AI Service Layer Migration — Postmortem & Forward Requirements

**Scope of this document:** `services/ai/` (the shared AI-call layer, GOVERNANCE.md §2.3)
and its callers, starting with `airflow/agents/recipe_agents.py` (Work Order #12).

**Status as of this document:** WO#12 complete, including a set of post-migration
amendments agreed in the same review session. Two callers (`job_agents.py` per
WO#11, `recipe_agents.py` per WO#12) are now fully migrated. Several more
callers and two more providers remain — see Part 4.

**Purpose:** This document exists so a reviewer can confirm WO#12 passed, and so
whichever agent picks up the next work order (WO#13+) has one place to find
what's already done, what was deliberately deferred and why, and what the
end-state cleanup looks like once every caller has migrated. Part 4 in
particular is written to be consumed as a requirements list, not just a
narrative.

---

## Part 0 — Chain of migrations so far

| Work Order | Caller migrated | Provider work | Status |
|---|---|---|---|
| WO#11 | `job_agents.py` | Built `services/ai/base.py`, `services/ai/keys.py`, `services/ai/providers/gemini.py` (`call_gemini_text`, `call_gemini_json`, `MODEL_FLASH`, `MODEL_FLASH_LITE`) | ✅ Complete (foundation this document builds on; not re-verified here) |
| WO#12 | `airflow/agents/recipe_agents.py` | Added `call_gemma_json`, `MODEL_GEMMA` to `providers/gemini.py` | ✅ Complete, with amendments — this document |
| WO#13 (planned, not started) | `weekly_agents.py`, possibly `workout_plans.py` / `media_recommend.py` | None (reuses existing Gemini-shaped functions) | Not started |
| Future, unscheduled | `blog_agents.py` (`_groq_llama`, `_cerebras`), `finance_upload.py` | New `providers/groq.py`, `providers/cerebras.py`, SDK-call design discussion | Not started |
| Future, unscheduled | Vision support | New vision function(s) in `services/ai/` | Not started |
| Future, unscheduled | Provider-agnostic wrapper (GOVERNANCE.md §2.3 target) | `call_ai_text()` / `call_ai_json()` | Deferred — needs ≥2 real providers to generalize from |

---

## Part 1 — What WO#12 delivered, as originally scoped

This is the mechanical migration exactly as the work order specified it, before
any follow-up changes. Recorded here so the "originally scoped" baseline is
unambiguous for a reviewer.

1. Added `call_gemma_json(prompt, model=MODEL_GEMMA)` and `MODEL_GEMMA` to
   `services/ai/providers/gemini.py`, extracted from `recipe_agents.py`'s
   private `_gemma()`. Built on the shared `post_with_retry` helper, same as
   `call_gemini_text`/`call_gemini_json`.
2. Re-exported `call_gemma_json` and `MODEL_GEMMA` from `services/ai/__init__.py`.
3. In `recipe_agents.py`:
   - Replaced the three `_gemini_flash_json(...)` call sites
     (`agent_extract_recipe`, `agent_discover_recipes_pantry`,
     `agent_discover_recipes_open`) with `call_gemini_json(...)`.
   - Replaced the one `_gemma(prompt)` call site (`agent_normalize_ingredients`)
     with `call_gemma_json(prompt)`.
   - Deleted the now-dead private `_gemini_flash()`, `_gemini_flash_json()`,
     and `_gemma()` definitions.
   - Kept `_gemini_key()`, used only by `agent_extract_recipe_from_image()`,
     which was explicitly out of scope (raw vision payload doesn't fit
     `call_gemini_text`/`call_gemini_json`'s signatures).
4. Confirmed zero changes required in `domains/recipes/routers/recipe_extract.py`,
   `domains/recipes/routers/recipe_discovery.py`, and `services/recipe_service.py`
   — all three call only public `agent_*` functions, never the private helpers.

### Original acceptance criteria (as verified against a mocked HTTP layer)

| # | Criterion | Result |
|---|---|---|
| 1 | `agent_extract_recipe()` produces an identical request via `call_gemini_json` | ✅ |
| 2 | `agent_normalize_ingredients()` produces an identical request via `call_gemma_json` (no `systemInstruction`, no `responseSchema`, `responseMimeType: application/json` present) | ✅ |
| 3 | `agent_discover_recipes_pantry()` / `agent_discover_recipes_open()` route through `call_gemini_json` correctly | ✅ |
| 4 | `agent_extract_recipe_from_image()` is **completely unchanged** (byte-identical `git diff`) | ✅ at time of WO#12 — **superseded in Part 2, amendments 4/5/6 below; see Reviewer Note** |
| 5 | `_gemini_flash`, `_gemini_flash_json`, `_gemma` no longer defined in `recipe_agents.py` | ✅ |
| 6 | `_gemini_key()` still defined, used only by the vision function | ✅ at time of WO#12 — **superseded in Part 2, amendment 5; `_gemini_key()` no longer exists at all** |
| 7 | Callers (`recipe_extract.py`, `recipe_discovery.py`, `recipe_service.py`) need zero changes | ✅ |
| 8 | `grep` confirms full removal of `_gemini_flash`/`_gemini_flash_json` | ✅ |

Two of the original eight criteria (4 and 6) were later intentionally superseded
by amendments agreed in the same review pass — see Part 2. This is expected and
should not be read as a regression; the amendments were explicit, scoped
decisions, not drift.

---

## Part 2 — Post-migration amendments (agreed after the initial migration)

The first five items below were raised in review immediately after WO#12's
mechanical migration was delivered and verified, and approved before
implementation. A sixth was added in a follow-up round of the same overall
review session, after reconsidering whether one of the five (amendment 3,
a flagged-but-deferred item) could reasonably be closed out immediately
instead of left for a future work order — see Amendment 6. All six are
documented separately from Part 1 because a reviewer checking WO#12 against
its original acceptance criteria needs to know these were deliberate,
in-session amendments — not scope creep discovered later, and not a failure
to follow the original work order.

### Amendment 1 — `call_gemma_json` gained a `retries` parameter

**Before:** `call_gemma_json(prompt: str, model: str = MODEL_GEMMA)` — no way
for a caller to override retry count, unlike `call_gemini_text`/`call_gemini_json`
which both expose `retries: int = 3`.

**After:** `call_gemma_json(prompt: str, model: str = MODEL_GEMMA, retries: int = 3) -> str`.

**Why:** API symmetry across the three `services.ai.providers.gemini` functions.
No caller currently overrides it (recipe_agents.py uses the default), so this
is additive and behavior-preserving for existing call sites.

**Verified:** Mocked a sustained 429 response and confirmed `call_gemma_json(..., retries=2)`
attempts exactly twice before raising, matching `post_with_retry`'s documented
contract.

### Amendment 2 — `services/ai/__init__.py` docstring updated

**Before:** Docstring described the module's contract as "THIS work order
(WO#11)" and didn't mention Gemma/`call_gemma_json` at all, despite the module
now exporting them.

**After:** Docstring now describes the WO#11 contract as established history,
adds a paragraph explaining the WO#12 Gemma addition and why it's a separate
function rather than a flag on `call_gemini_json`, and points future readers
at this postmortem document for the running status of the larger migration.

**Why:** A docstring that says "THIS work order" about a work order that
finished, sitting above code from a later work order, actively misleads the
next reader about what's current. Pure documentation fix, zero behavior change.

### Amendment 3 — Retry-inconsistency in `recipe_agents.py` flagged, not fixed *(at the time — see Amendment 6, this was resolved later in the same session)*

**Not a code change, at the time this was written.** Recorded here per
explicit request to carry this into the postmortem for future work-order
planning. **Update: the retry gap itself was closed the same session — see
Amendment 6 below. This section is left intact as the historical record of
why it wasn't fixed immediately, since the reasoning still matters for
future work (the vision call still isn't on a real service-layer function —
only its retry behavior was fixed).**

**The issue:** Before WO#12, all four AI calls in `recipe_agents.py` had zero
retry logic — a consistent (if fragile) posture. After WO#12's mechanical
migration, three of the four calls (`agent_extract_recipe`,
`agent_discover_recipes_pantry`, `agent_discover_recipes_open`, and
`agent_normalize_ingredients`) auto-retry on 429/503 via `post_with_retry`,
while `agent_extract_recipe_from_image()` — the one call left on a raw
`requests.post` — still fails immediately on a transient rate limit or
service hiccup.

**Why not fixed now:** Fixing it means either (a) migrating the vision call
onto the service layer (needs vision support to be designed and built there
first — explicitly out of scope for WO#12, see Part 4 item 4b), or (b) bolting
ad hoc retry logic onto the raw vision call as a stopgap, which would
duplicate `post_with_retry` outside the service layer — the exact pattern
this whole migration exists to eliminate. Neither is a WO#12-sized change.

**Disposition (superseded):** Originally accepted as known tech debt,
deferred to full vision migration. **Resolved same session — see Amendment 6.**
The remaining, still-deferred piece (migrating the vision call onto a real
service-layer function rather than a raw request routed through the shared
retry primitive) is unchanged and tracked in Part 4, item 4b.

### Amendment 6 — Vision call now retries on 429/503, via `post_with_retry` directly

**The gap this closes:** After WO#12's mechanical migration (Part 1) and
amendments 4–5 above, `agent_extract_recipe_from_image()` was the only call
in `recipe_agents.py` still on a bare `requests.post` with no retry
protection — everything else auto-retries on 429/503 via `post_with_retry`.
Fully closing this the "proper" way means designing real vision support into
`services/ai/` (a new `call_gemini_vision_json`-style function), which
remains explicitly out of scope (Part 4, item 4b) — that's real new-capability
design, not a mechanical change.

**What was actually done instead:** `post_with_retry` itself is fully
provider-agnostic and vision-agnostic — it just POSTs a `url`/`payload` and
retries on 429/503. The only reason the vision call couldn't already use it
directly was a hardcoded `timeout=90` inside `post_with_retry`, while the
vision call needs `timeout=120` for its larger multimodal payload. So:

1. **`services/ai/base.py`**: `post_with_retry` gained an optional
   `timeout: float = 90.0` keyword parameter. The default exactly matches
   what was previously hardcoded, so `call_gemini_text`, `call_gemini_json`,
   and `call_gemma_json` — none of which pass `timeout` explicitly — are
   completely unaffected; they still get 90s exactly as before.
2. **`recipe_agents.py`**: `agent_extract_recipe_from_image()` now calls
   `post_with_retry(url, payload, retries=3, provider_name="Gemini",
   resource_label=MODEL_FLASH, timeout=120)` instead of a bare
   `requests.post(url, json=payload, timeout=120)`. The `timeout=120` is
   passed explicitly, so the original 120s budget for vision calls (vs. 90s
   for everything else) is preserved exactly. The response-parsing logic
   simplified accordingly — `post_with_retry` already returns the parsed
   JSON body and already calls `resp.raise_for_status()` internally, so the
   function's own `resp.raise_for_status()` / `resp.json()` calls were
   removed; the broad `except Exception` fallback (returning
   `{"title": "Recipe from Image", "raw_ingredient_lines": []}`) is
   unchanged.
3. `post_with_retry` is imported directly from `services.ai.base` in
   `recipe_agents.py` — **an intentional exception** to the usual convention
   of callers importing only from the top-level `services.ai` package.
   `post_with_retry` is a low-level primitive, not part of the public
   provider-function contract `services/ai/__init__.py` re-exports; it's
   being reused directly here only because no vision-specific provider
   function exists yet to wrap it (see item 4b, Part 4). When real vision
   support lands in `services/ai/`, this direct import should go away in
   favor of whatever `call_gemini_vision_json`-equivalent replaces it — flag
   this as a small piece of item 4b's future scope, not a new permanent
   pattern to imitate elsewhere.
4. The `import requests` at the top of `recipe_agents.py` was removed —
   after this change, nothing in the file calls `requests.*` directly
   anymore.

**Why the vision-specific payload construction (the `inlineData` block) was
NOT touched:** That part is genuinely vision-specific and is exactly the
piece that still needs real design work before it belongs in `services/ai/`
(schema shape for image+text multimodal requests, whether/how to generalize
beyond Gemini's vision format, etc.). Reusing `post_with_retry` only
addresses the transport/retry layer, which was always fully generic — it
never needed vision-specific work, just a missing parameter.

**Verified:**
- Mocked a 429-then-200 sequence and confirmed
  `agent_extract_recipe_from_image()` now retries once and recovers,
  where it previously would have failed immediately on the first 429.
- Confirmed the retry preserves `timeout=120` on every attempt (not the
  90s default).
- Confirmed the three already-migrated call sites
  (`call_gemini_text`/`call_gemini_json`/`call_gemma_json` users) still
  receive `timeout=90.0` unchanged, since none of them pass the new
  parameter.
- Confirmed the exhausted-retries fallback still returns the exact original
  stub dict `{"title": "Recipe from Image", "raw_ingredient_lines": []}`.

**Scope note:** Like amendments 4–5, this further diverges
`agent_extract_recipe_from_image()` from its original WO#12 pre-migration
baseline — now on top of the model-constant and key-lookup changes, its
request/retry handling is fully routed through the shared primitive. The
function's *payload construction* (system instruction, inlineData, schema)
remains completely untouched and vision-specific, as intended.

### Amendment 4 — `agent_extract_recipe_from_image()` now references `MODEL_FLASH`

**Before:** Hardcoded the literal string `"gemini-2.5-flash"` directly in the
URL-building f-string — the only place in `recipe_agents.py` that didn't
reference a shared constant, because it predates `services/ai/` entirely.

**After:**
```python
url = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{MODEL_FLASH}:generateContent?key={get_provider_key('gemini')}"
)
```
`MODEL_FLASH` imported from `services.ai` (already the model ID used by
`agent_extract_recipe`'s `call_gemini_json` call, so the vision function was
already implicitly required to stay in sync with it by hand — now it can't
drift).

**Why:** If/when this model ID changes (including a Google-side sunset of
`gemini-2.5-flash`), every other call site updates by editing one constant;
this one previously would not have, silently continuing to call a retired
model until someone noticed the vision path specifically breaking.

**Scope note:** WO#12's original acceptance criterion required
`agent_extract_recipe_from_image()` to be byte-identical to its pre-migration
form. This amendment explicitly and intentionally breaks that criterion, with
approval, in the same review session the criterion was verified. See Reviewer
Note below.

### Amendment 5 — `_gemini_key()` removed entirely

**Before:** `recipe_agents.py` retained a private `_gemini_key()` helper
(`os.environ.get("GEMINI_API")`), kept alive solely because the vision
function used it — duplicating exactly what `services.ai.keys.get_provider_key("gemini")`
already does for every other call in the file.

**After:** `_gemini_key()` deleted. `agent_extract_recipe_from_image()` calls
`get_provider_key("gemini")` (imported from `services.ai.keys`) directly. The
now-unused `import os` was also removed from the top of the file — nothing
else in `recipe_agents.py` used `os`.

**Why:** Once amendment 4 was being made to the same function anyway, leaving
a duplicate key-lookup helper around for one caller with no other users made
no sense. `get_provider_key("gemini")` is behavior-identical to the deleted
`_gemini_key()` for every real deployment (see `services/ai/keys.py`'s
docstring for the byte-for-byte equivalence argument already established in
WO#11).

**Scope note:** Same as amendment 4 — this further diverges
`agent_extract_recipe_from_image()` from its WO#12 pre-migration baseline,
with approval.

### Reviewer note on amendments 4, 5, and 6

WO#12's acceptance criterion #4 ("`agent_extract_recipe_from_image()` is
completely unchanged, zero-diff") **no longer holds**, by design. The function
now differs from the pre-WO#12 baseline in three respects:

1. `"gemini-2.5-flash"` literal → `MODEL_FLASH` constant (amendment 4)
2. `_gemini_key()` call → `get_provider_key("gemini")` call (amendment 5)
3. Raw `requests.post(...)` + manual `raise_for_status()`/`resp.json()` →
   `post_with_retry(..., timeout=120)`, giving the call 429/503 retry
   protection it never had before (amendment 6)

Amendments 4 and 5 are import-and-constant substitutions only — the
resulting HTTP request (URL, payload shape, headers, timeout) was
unaffected. Amendment 6 is a genuine behavior change, but a scoped and
verified one: the request sent on the *first* attempt is byte-identical to
before (same URL, same payload, same `timeout=120`); what changed is that a
429/503 response now triggers the same backoff-and-retry behavior every
other call in this file already has, instead of failing immediately. No
prompt, schema, or model was altered, and the vision-specific payload
construction (`inlineData`, system instruction) is completely untouched —
only the transport/retry wrapper around it changed.

**For sign-off purposes:** treat the original criterion #4 as replaced by:
*"`agent_extract_recipe_from_image()`'s outbound HTTP request (URL, payload,
timeout) is unchanged on the first attempt; its source code now references
shared constants/helpers instead of local duplicates (amendments 4–5), and
its transport layer now retries on 429/503 like every other call in the
file (amendment 6) — a deliberate, scoped exception to 'zero behavior
change,' verified against a mocked HTTP layer."* Verified ✅ — see the
verification notes under each amendment above.

---

## Part 3 — Accepted tech debt / deliberately deferred (full list)

Consolidated from both the original WO#12 hard boundaries and the amendment
session. Nothing in this list is a bug — each is a scoped decision to not act,
with a reason.

| Item | Why deferred | Where it's tracked to be resolved |
|---|---|---|
| ~~Vision call has no retry-on-429/503 while every other call in the file now does~~ **RESOLVED (Amendment 6)** | Was deferred pending full vision-service-layer migration; resolved instead by adding an optional `timeout` param to `post_with_retry` and routing the existing raw vision request through it directly — no vision-specific abstraction needed for just the retry/transport layer | Closed. Vision call now retries on 429/503 identically to every other call in the file. |
| `agent_extract_recipe_from_image()` still builds its own request and payload directly, not through a `call_gemini_json`-style service-layer function | Vision payload shape (`inlineData`) doesn't fit existing signatures; designing multimodal support is new-capability work, not a mechanical port. Amendment 6 fixed the *transport/retry* layer only — the payload-construction layer is unchanged and still needs real design work | Part 4, item 4b |
| `call_gemma_json` retries on 429/503; the original `_gemma()` had no retry logic at all | Explicit instruction in WO#12's Step 1 to build on the shared retry helper, for consistency with `call_gemini_text`/`call_gemini_json`; judged an acceptable, intentional behavior change rather than a strict "same logic" port | No further action planned — flagged for awareness only. Revisit only if it causes observed problems (e.g. unexpectedly long hangs on a 429 storm during ingredient normalization) |
| Groq (`_groq_llama`) and Cerebras (`_cerebras`) in `blog_agents.py` not migrated | Genuinely different provider call shapes (different retry semantics, Cerebras returns a tuple); no `providers/groq.py` or `providers/cerebras.py` exist yet | Part 4, item 6 (provider modules), item 7 (wrapper) |
| `finance_upload.py`'s SDK-based Gemini call not migrated | Different calling convention entirely (SDK vs. raw HTTP); needs a design discussion before a mechanical port makes sense | Not yet scheduled — flag for scoping discussion before assigning a work order |
| Provider-agnostic `call_ai_text()`/`call_ai_json()` wrapper (GOVERNANCE.md §2.3 target) not built | Needs ≥2 real, meaningfully-different providers to generalize from correctly; Gemini alone risks guessing the wrong config shape | Part 4, item 7 |

---

## Part 4 — Requirements checklist for after all migrations are complete

This section is written for the agent(s) executing future work orders. Each
item below should become either part of a future work order's scope or an
explicit "still open" note carried forward into the next postmortem revision.
Do not close this document out / mark the migration fully finished until every
item below is either done or explicitly re-deferred with a reason.

- [ ] **1. Retire `airflow/agents/gemini_client.py`.** This is the original
  monolith `services/ai/base.py` and `providers/gemini.py` were extracted
  from (per WO#11). Once every caller that still imports from it directly has
  migrated, confirm via `grep -rn "gemini_client" --include=*.py .` across the
  whole repo, and if the only remaining references are inside
  `gemini_client.py` itself, delete the file (or formally deprecate it with a
  clear docstring redirect if something outside this repo's control still
  depends on its import path).

- [ ] **2. Audit every remaining agent module for hardcoded model-ID literals**
  and replace them with the shared `MODEL_FLASH` / `MODEL_FLASH_LITE` /
  `MODEL_GEMMA` constants from `services.ai`, mirroring the fix applied to
  `agent_extract_recipe_from_image()` in Part 2, amendment 4 of this document.
  *(Note on scope of this item: no file literally named `models.py` was found
  within the AI-service-layer migration's scope — `domains/recipes/models.py`
  is the unrelated SQLAlchemy ORM layer for the recipes domain. This item is
  the intended generalization of that instruction: eliminate duplicated
  literal model-ID strings anywhere they still exist outside `services/ai/`,
  the same class of problem amendment 4 fixed for the vision function. If a
  genuinely different file was meant, flag it back to this document's
  maintainer for correction.)* Known candidates, unverified until each is
  actually opened:
  - [ ] `blog_agents.py` — check `_groq_llama`, `_cerebras`, and any Gemini
    calls in this file for literal model strings.
  - [ ] `weekly_agents.py` — its `_gemini_flash_json` migration (WO#13,
    planned) should include this check as part of that work order, not as an
    afterthought.
  - [ ] `workout_plans.py`'s `_call_gemini_for_plan`.
  - [ ] `media_recommend.py`'s `_gemini_explain`.
  - [ ] `finance_upload.py`'s SDK-based call (check whether the SDK client
    takes a model-ID string the same way, or a different configuration shape
    entirely — don't assume the pattern transfers).

- [ ] **3. Audit every remaining agent module for private per-file key-lookup
  helpers** duplicating `services.ai.keys.get_provider_key()` (the
  `_gemini_key()` pattern just fully removed from `recipe_agents.py` per
  amendment 5). Remove each as its owning module migrates. Do not leave one
  behind "just for one caller" the way `recipe_agents.py` did between WO#12's
  initial delivery and its amendment session — that gap is exactly what
  amendment 5 closed, and it's cheap to close immediately in future
  migrations rather than revisiting later.

- [x] ~~4a. Give the vision call retry coverage~~ **DONE — Amendment 6.**
  `agent_extract_recipe_from_image()` now routes through the shared
  `post_with_retry` (imported directly from `services.ai.base`, with an
  explicit `timeout=120` override) instead of a bare `requests.post`. No
  further action needed on this specific point.

- [ ] **4b. Design and build real multimodal/vision support in
  `services/ai/`** — a proper provider-level function (e.g.
  `call_gemini_vision_json(system, image_base64, mime_type, prompt, schema)`
  in `providers/gemini.py`) that owns the `inlineData` payload construction
  itself, the way `call_gemini_json` owns its payload construction today.
  Once it exists, migrate `agent_extract_recipe_from_image()` onto it and
  remove its direct `from services.ai.base import post_with_retry` (that
  import was always meant as a stopgap — see amendment 6's note 3 — not a
  pattern to leave in place long-term). This remains real new-capability
  design work, not a mechanical port: figure out whether/how to generalize
  the multimodal shape beyond Gemini's specific vision format before
  building it, the same way the provider-agnostic wrapper (item 7) needed
  ≥2 real providers before it could be designed correctly.

- [ ] **5. Re-audit retry coverage across every migrated module**, not just
  `recipe_agents.py`. Amendment 6 closed the one known gap in
  `recipe_agents.py`, but that check hasn't been run against any other
  agent module (`weekly_agents.py`, `workout_plans.py`, `media_recommend.py`,
  `blog_agents.py`, `finance_upload.py`) — do so as each migrates. Don't
  assume; re-run a grep/audit pass similar to the one done for this
  document's Part 1, item 4/8, and Amendment 6's verification.

- [ ] **6. Build `services/ai/providers/groq.py` and
  `services/ai/providers/cerebras.py`.** Groq's current caller
  (`_groq_llama` in `blog_agents.py`) has no retry logic today. Cerebras's
  (`_cerebras`, also in `blog_agents.py`) has materially different behavior:
  honors a `Retry-After` header, uses a different backoff schedule (75/150/
  300/600s), and returns a `(content, remaining_tokens)` tuple rather than
  just content. Do not force either through `post_with_retry` unmodified —
  determine whether `post_with_retry` needs new optional parameters
  (`Retry-After` support, configurable backoff schedule, alternate return
  shape) or whether these providers need their own retry helper. This
  decision was explicitly deferred by both WO#11 and WO#12 pending real
  provider shapes to generalize from — this is the point where that
  generalization should finally happen.

- [ ] **7. Design and build the provider-agnostic `call_ai_text()` /
  `call_ai_json()` wrapper** described in GOVERNANCE.md §2.3, once items 6
  (Groq/Cerebras providers) exist. This is the last major piece — with
  Gemini, Gemma, Groq, and Cerebras all represented, there will finally be
  enough real call-site shapes to generalize a provider-selection/config
  interface from with confidence, rather than guessing from Gemini alone (the
  reason this was deferred at both WO#11 and WO#12).

- [ ] **8. Update `services/ai/__init__.py`'s docstring again** once the
  provider-agnostic wrapper (item 7) lands. It currently documents a
  Gemini-plus-Gemma-only contract (updated per amendment 2 of this document);
  that framing will go stale the moment a provider-agnostic entry point is
  added, the same way the WO#11-only framing went stale between WO#11 and
  WO#12. Make updating this docstring an explicit step in whatever work order
  ships the wrapper, not an afterthought — this is the second time it's had
  to be corrected for staleness.

- [ ] **9. Update GOVERNANCE.md §2.3 itself** once the service layer reaches
  its intended final shape, so the governance doc and the code it describes
  don't drift apart. (This document doesn't check GOVERNANCE.md's current
  wording — flag it for whichever work order finishes the migration.)

- [ ] **10. Re-verify the "zero caller changes required" claim fresh for
  each module as it migrates** — don't carry forward an assumption from this
  document or from a work order's own text. WO#12's text assumed
  `_gemini_flash()` had call sites in `recipe_agents.py` to update; it
  didn't — the function was already dead code. Always grep the actual call
  sites of the private helper being removed before assuming a caller list is
  complete or that "public API only" callers truly never touch a private
  helper.

- [ ] **11. Once every caller is migrated, revisit whether
  `post_with_retry`'s retry schedule** (429: wait `30 * (attempt+1)`s; 503:
  wait `5 ** attempt`s) **should become configurable per call** rather than
  hardcoded in `services/ai/base.py`. More callers with potentially different
  latency tolerances (a batch Airflow DAG task vs. a user-facing FastAPI
  request handler, for instance) will depend on this shared schedule by then;
  worth confirming it's still the right one-size-fits-all default before
  treating it as permanent.

---

## Appendix — Verification method used throughout

No live API access was available while producing this document or its
underlying code changes. All request-construction claims (Part 1's
acceptance criteria, Part 2's amendment verifications) were checked by
monkey-patching `requests.post` with a fake response and inspecting the
captured `url`/`payload`/`timeout` for each call path, then asserting against
the pre-migration baseline's known request shape. This is noted per the
governing work order's instruction to flag live-API substitutions explicitly
(⚠️) rather than silently assume equivalence. If live API verification
becomes available before this migration is considered fully closed out, it
should be run at least once against the real Gemini endpoint to confirm the
mocked assumptions (payload field names, schema enforcement behavior) still
hold.
