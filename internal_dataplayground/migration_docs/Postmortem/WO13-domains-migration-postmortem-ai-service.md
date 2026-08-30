# WO#13 (Rewritten) — Postmortem & Post-Migration Requirements

**Scope of this document:** AI Service Layer batch migration (`weekly_agents.py`,
`workout_plan_ai_generator.py`, `media_recommend.py`, `blog_agents.py`'s
Gemini functions) + TMDB duplication reconciliation, and every follow-on
change made in the same engagement after WO#13 itself closed.

**Audience:** the next agent or engineer picking up work in this codebase —
WO#14, WO#15, WO#16, a GOVERNANCE.md update pass, or a general cleanup
sweep. Treat this as the authoritative account of what changed, why, what
was verified, and what is still open.

**Read Part A and Part B as distinct.** Part A is what WO#13's own text
authorized and how it was executed against its own rules. Part B is
follow-on work a human reviewer explicitly requested in the same
conversation, *after* WO#13 closed — some of it directly contradicts
WO#13's own HARD BOUNDARIES (e.g. "do not reconcile" the Task 2 drift),
and that contradiction is intentional and authorized, not an error. A
reviewer checking "was WO#13 done correctly" should evaluate Part A
against WO#13's text, and Part B as a separately-authorized, separately
verified set of changes layered on top.

---

## Executive Summary

WO#13 (Rewritten) migrated five call sites off bespoke, duplicated
Gemini-calling code and onto the shared `services/ai/` layer
(GOVERNANCE.md §2.3), while keeping backward compatibility with the two
already-migrated callers from WO#11 and WO#12. It also ran a required
behavioral diff between the two independent TMDB-calling implementations
(`services/tmdb_service.py` and `airflow/agents/media_agents.py`) and,
per its own rules, found and *reported* a drift without fixing it.

After WO#13 closed, the requester asked follow-up questions that led to
five additional fixes, all in `airflow/agents/media_agents.py`,
`services/ai/keys.py`, `domains/workout/routers/workout_plan_ai_generator.py`,
and `domains/media/routers/media_recommend.py`. These were explicitly
authorized turn-by-turn in conversation and are documented in Part B.

**Net result:** 12 files now differ from their pre-engagement state (see
full inventory below). `services/tmdb_service.py` was read for comparison
only and was never edited. Three `services/ai/` files
(`base.py`, `keys.py`, `__init__.py`) are **reconstructed stubs** — they
were never provided as source material in this engagement and must be
checked against the real files before any of this is merged. This is the
single most important open item in this document — see "Known Caveats"
below.

---

## Part A — Executed Under WO#13 (Rewritten)'s Own Authority

### Task 1: AI Service Layer batch migration

**Files edited (in WO#13's own SCOPE list):**

| File | Change |
|---|---|
| `services/ai/providers/gemini.py` | `call_gemini_json(system, prompt, schema, model, retries)` → `call_gemini_json(prompt, schema=None, system=None, model=MODEL_FLASH, retries=3)`. `schema`/`system` now optional; omitted `system` means no `systemInstruction` is sent at all (not an empty one). |
| `airflow/agents/weekly_agents.py` | `agent_plan_meals()` and `agent_schedule_workouts()` migrated from a bespoke `_gemini_flash_json()` to `call_gemini_json()`. `_gemini_flash_json()` and `_gemini_key()` deleted. Explicit `model=MODEL_FLASH` added at both call sites (model-literal spot-check). |
| `domains/workout/routers/workout_plan_ai_generator.py` | `_call_gemini_for_plan()` kept as a thin wrapper, now delegates to `call_gemini_json(prompt, schema=None, system=system, model=MODEL_FLASH)`. Dead `os`/`requests` imports removed. |
| `domains/media/routers/media_recommend.py` | `_gemini_explain()` migrated to `call_gemini_json(prompt, schema=None, system=None, model=MODEL_FLASH)` — explicit `system=None` preserved as an intentional detail (this caller never sent a system instruction). |
| `airflow/agents/blog_agents.py` | Grepped first per WO's own instruction; confirmed exactly 4 live call sites (`agent_readme_writer`, `agent_researcher`, `agent_editor`, `agent_idea_expander`) — matched the WO's inferred list exactly, no discrepancy. All 4 migrated to `call_gemini_text()`/`call_gemini_json()`. `_gemini_flash()`, `_gemini_flash_json()`, `_gemini_key()` deleted. Groq logic (`_groq_llama`, `_groq_key`) and Cerebras logic (`_cerebras`, `_cerebras_key`, `_CEREBRAS_BACKOFF`, `_CEREBRAS_QWEN3`, `_CEREBRAS_LLAMA33`, and the commented-out prior `_cerebras()` block) confirmed untouched — verified via grep, all 5 Groq/Cerebras call sites still present. Module docstring routing table and section-header comments confirmed untouched. |
| `airflow/dags/life_os_weekly_synthesis.py` | `from agents.blog_agents import _gemini_flash` → `from services.ai import call_gemini_text`, call site updated to match. Landed in the same change as the `blog_agents.py` edit, per WO instruction. |

**Files edited beyond the literal SCOPE list — flagged, as required by the
WO's own OUTPUT FORMAT instruction:**

| File | Change | Why this was in-bounds despite not being in SCOPE |
|---|---|---|
| `airflow/agents/job_agents.py` | One call site (`score_job_batch()`) converted from positional to explicit-keyword arguments. | WO#13 Step 1 explicitly required this: *"Update the WO#11 and WO#12 call sites to use explicit keyword arguments if they currently rely on positional order the new signature would break."* Without this, `call_gemini_json`'s new argument order would have silently misassigned this call site's arguments. |
| `airflow/agents/recipe_agents.py` | Three call sites (`agent_extract_recipe`, `agent_discover_recipes_pantry`, `agent_discover_recipes_open`) converted the same way. | Same reasoning. `call_gemma_json` and the vision function's direct `post_with_retry` call were confirmed untouched — those are explicitly out of scope (WO#16). |

**A note on `weekly_agents.py` specifically:** the copy of this file
available at the start of the engagement had *not* yet received the
WO#10 month-boundary fix (`date(year, month, day + offset)` → 
`week_start + timedelta(days=d - 1)`) — it still had the buggy
raw-arithmetic version. Per the rewritten WO#13's own explicit
instruction not to "helpfully revert" a fix or proceed on a stale
snapshot, that fix was applied as an isolated, separately-attributed
edit *before* WO#13's own Step 2 substitution, and the two edits were
kept diff-separable. The requester later uploaded the actual current
file, which confirmed the fix was applied identically to the real
upstream version (one incidental docstring whitespace difference from
transcription was found and corrected to match).

### Task 2: TMDB duplication reconciliation

**Read (comparison source, never edited):** `services/tmdb_service.py`

**Compared:** `airflow/agents/media_agents.py`'s `get_tmdb_watch_providers()`
against `services/tmdb_service.py`'s `get_streaming_providers()`, on the
three axes WO#13 specified.

**Finding: disagreement**, on two of three axes:
1. US/flatrate scoping — agreed.
2. 404 handling — **disagreed.** `tmdb_service.py` treats a 404 as
   "confirmed no availability" (`return []`). `media_agents.py` called
   `resp.raise_for_status()` unconditionally, so a 404 raised
   `requests.HTTPError`. Confirmed concretely via a simulated 404
   response.
3. Provider-ID extraction — minor difference (`media_agents.py`
   defensively filters entries missing `provider_id`; `tmdb_service.py`
   doesn't). Not flagged as drift requiring action — more defensiveness
   isn't a bug.

**Per WO#13's own explicit HARD BOUNDARY** ("If the diff... finds
behavioral drift... do not silently reconcile it... mark the criterion
⚠️, and let it be ticketed separately"), **this was reported and left
unfixed within WO#13 itself.** `media_agents.py` received zero edits
under WO#13's own authority. No cross-referencing "confirmed equivalent"
comment was added (Step 9 only applies on full agreement, which this
wasn't).

**A downstream consequence was also identified and documented** (not
fixed under WO#13): `life_os_refresh_streaming_availability.py`'s
`task_select_and_refresh()` catches any exception from
`get_tmdb_watch_providers()` as "transient failure, retry later," and
does not advance `streaming_fetched_at`. Because of the 404 drift above,
a title TMDB legitimately 404s on would be retried forever instead of
being recorded as "confirmed: no availability" — the exact case the
`life_os_refresh_streaming_availability` DAG exists to handle correctly.

### Verification performed under WO#13's own authority

- **Backward compatibility**: simulated the exact payload shape both
  pre- and post-migration for `job_agents.py::score_job_batch()` and all
  three `recipe_agents.py` call sites — confirmed byte-identical JSON.
- **New migrations' payload shape**: simulated and confirmed
  `weekly_agents.py` (both `systemInstruction` and `responseSchema`
  present), `workout_plan_ai_generator.py` (`systemInstruction` present,
  no `responseSchema`), `media_recommend.py` (neither present, only
  `responseMimeType`).
- **WO#10 fix regression check**: swept all 52 Mondays of 2026 against
  the stale raw-arithmetic version (11/52 crashed, 21.2% — matching the
  postmortem's cited "~21%" figure) and against the `timedelta` version
  (0/52 crashed). Confirmed the requester's uploaded ground-truth file
  matched the fix exactly after applying WO#13's own Step 2 substitution
  on top of it (diffed byte-for-byte).
- **`blog_agents.py` call-site confirmation**: grepped for
  `_gemini_flash(` and `_gemini_flash_json(`, found exactly the 4
  expected functions, no discrepancy from the WO's inferred list.
  Grepped again post-edit to confirm zero remaining references and that
  all 5 Groq/Cerebras call sites are untouched.
- **404 drift demonstration**: simulated a 404 response through both
  implementations' logic, confirmed `tmdb_service.py` → `[]`,
  `media_agents.py` (pre-fix) → raises.
- **DAG import audit**: reviewed import statements in all 8 other DAGs
  that call into `blog_agents.py`'s public functions
  (`life_os_blog_creator.py`, `life_os_blog_finalizer.py`,
  `life_os_blog_scout.py`, `life_os_idea_expander.py`,
  `life_os_readme_writer.py`, `life_os_code_narrate.py`,
  `life_os_code_comment.py`, `life_os_code_improve.py`) — confirmed none
  import `_gemini_flash`/`_gemini_flash_json` directly; only
  `life_os_weekly_synthesis.py` did.
- **Marked ⚠️, not verified**: true end-to-end router verification
  (`POST /workout/plans/generate`, `POST /media/recommend/generate`)
  against a live FastAPI/SQLAlchemy/DB stack — not available in this
  environment. Call-site preservation (the router still calls the same
  function name/signature) plus payload-shape verification were used as
  a documented substitute, per WO#13's own allowance for this situation.

### Acceptance criteria — final state under WO#13's own text

All required criteria: ✅, except:
- Task 2's behavioral-comparison criterion: ⚠️ (disagreement found,
  reported, not reconciled — exactly the outcome the WO's own rules
  call for on disagreement, not a failure to meet the bar).
- End-to-end live router verification: ⚠️ (mocked/simulated substitute
  used, explicitly stated, per the WO's own allowance).

---

## Part B — Follow-On Changes (Requested After WO#13 Closed)

**None of the changes in this section were authorized by WO#13's own
text.** Several directly reverse an explicit WO#13 instruction (Task 2's
"do not reconcile" rule). They were made only because the requester
explicitly asked for them in conversation, turn by turn, after WO#13's
own deliverable was already complete and reported. A reviewer should
treat this section as a separate, human-authorized change set — not part
of what WO#13 itself would have produced unattended.

### Fix 1 — `media_agents.py`: 404 handling now matches `tmdb_service.py`

**What changed:** `get_tmdb_watch_providers()` now checks
`resp.status_code == 404` and returns `None` before calling
`raise_for_status()`, instead of letting a 404 raise
`requests.HTTPError`. This directly reverses WO#13's own "do not
reconcile" instruction for this specific finding — done because the
requester asked, after seeing the Task 2 finding, whether it could be
fixed.

**Scope discipline preserved:** the fix stays isolated to
`media_agents.py`. `services/tmdb_service.py` was not touched, and no
import was added in either direction — GOVERNANCE §2.2 (DAGs never
import `services/`) still holds.

**Verified:** simulated 404 (→ `None`, no exception), simulated 500
(→ still raises, unaffected), simulated 200 with real data (→ still
parses correctly, unaffected).

**Downstream effect:** `life_os_refresh_streaming_availability.py`
needed **no changes** — it already treats a `None` result as a valid,
non-error outcome (per its own existing comment), so this fix alone
closes the "retried forever" gap described in Part A.

### Fix 2 — `media_agents.py`: `media_type` now accepts `"tv_show"`

**What changed:** `get_tmdb_watch_providers(tmdb_id, media_type)` now
accepts `"movie"`, `"tv"`, or `"tv_show"` (previously only `"movie"`/`"tv"`,
raising `ValueError` on anything else). `"tv_show"` maps to the same
`"tv"` URL segment as `"tv"`, matching `tmdb_service.py`'s own public
contract and the actual value TMDB-sourced TV items carry
(`tmdb_service.py::search_tv()` sets `"media_type": "tv_show"` on every
normalized result it returns — this is the value that ends up in
`media_items.media_type`).

**Why this was flagged rather than pre-emptively fixed under WO#13:**
the one current caller (`life_os_refresh_streaming_availability.py`)
already converts to `"movie"`/`"tv"` before calling, so this was never a
live bug — only a footgun for a hypothetical future caller passing
`MediaItem.media_type` directly. Flagged as an "out of scope
recommendation," then fixed at the requester's explicit request.

**Verified:** confirmed the existing caller's behavior (`"movie"` →
`"movie"`, `"tv"` → `"tv"`) is unchanged; confirmed `"tv_show"` → `"tv"`
now works instead of raising; confirmed genuinely invalid input still
raises `ValueError`.

### Fix 3 — `media_agents.py`: stale module docstring caveat corrected

**What changed:** the module docstring's "diff this against the real
`services/tmdb_service.py` before trusting it in production" caveat —
phrased as a still-outstanding TODO — was replaced with a "DIFF STATUS"
section stating plainly that the diff has now happened (WO#13 Task 2),
what it found, and what was and wasn't subsequently fixed (Fix 1 and Fix
2 above; the provider-ID defensive-filtering difference was left as-is,
explicitly, as not worth "fixing away"). The original caveat text is
preserved below it for history rather than deleted outright.

**Why this matters for future readers:** without this update, the next
engineer to open this file would see language implying the diff still
needs to happen, when it already has — and would have no way to know,
from the file alone, that two of the three differences it warned about
have since been resolved.

### Fix 4 — `services/ai/keys.py`: fail loudly on a missing key

**What changed:** `get_provider_key()` now raises `RuntimeError` if the
resolved environment variable is unset or empty, instead of silently
returning `None`. This matches the pattern `media_agents.py`'s own
`_tmdb_key()` already used.

**CRITICAL CAVEAT — read this before acting on it:** `services/ai/keys.py`
is a **reconstructed stub**, not a file that was ever provided as source
material in this engagement (see "Known Caveats" below for the full
explanation). This fix was applied to the reconstruction only. **The
real `keys.py` must be checked independently** — it may already fail
loudly in some other way, or may have a reason for the silent-`None`
behavior that isn't visible from the call sites alone. Do not assume
this fix is "done" until the real file is inspected.

### Fix 5 — Event-loop-blocking fix in two async FastAPI routes

**What changed:** `services/ai/`'s HTTP layer is synchronous
(`requests`, including blocking `time.sleep()` on 503 retries). Two
`async def` FastAPI route handlers call into it directly:
`generate_plan()` in `workout_plan_ai_generator.py` (via
`_call_gemini_for_plan`) and `generate_recommendations()` in
`media_recommend.py` (via `_gemini_explain`). A blocking call inside an
async handler stalls the *entire* event loop for that worker — every
other in-flight request, not just the slow one — for the duration of the
call, worse under 503 retries (potentially tens of seconds).

This was **not introduced by WO#13** — both functions built their own
synchronous HTTP calls before the migration too. It's flagged here
because WO#13 happened to be the first time both were routed through one
shared layer, making the pattern visible and worth fixing once instead
of twice.

**Fix applied:** at each of the two call sites (not inside
`services/ai/` itself, to avoid disturbing the Airflow-side synchronous
callers), the blocking call is now wrapped in `asyncio.to_thread(...)`
and awaited:
- `workout_plan_ai_generator.py`: `raw_json = await asyncio.to_thread(_call_gemini_for_plan, prompt, system)`
- `media_recommend.py`: `raw = await asyncio.to_thread(call_gemini_json, prompt, schema=None, system=None, model=MODEL_FLASH)`

**Verified:** a standalone `asyncio` demonstration proved the pattern
actually keeps the event loop free — total elapsed time for a "slow
Gemini call" run concurrently with unrelated event-loop work was ~0.31s,
not the ~0.6s it would be if the blocking call serialized with the other
work.

**Important scope note:** only these two call sites were fixed, because
they were the only two `services/ai/`-calling async routes visible in
this engagement's file set. **This does not mean these are the only two
in the whole codebase** — see "Post-Migration Follow-Up Required" below.

---

## Full File Inventory

**Edited under WO#13's own authority (Part A):**
- `services/ai/providers/gemini.py`
- `airflow/agents/weekly_agents.py`
- `domains/workout/routers/workout_plan_ai_generator.py` *(also touched again in Part B, Fix 5)*
- `domains/media/routers/media_recommend.py` *(also touched again in Part B, Fix 5)*
- `airflow/agents/blog_agents.py`
- `airflow/dags/life_os_weekly_synthesis.py`
- `airflow/agents/job_agents.py` *(flagged minimal edit, required for backward compat)*
- `airflow/agents/recipe_agents.py` *(flagged minimal edit, required for backward compat)*

**Edited under Part B's explicit follow-on authorization:**
- `airflow/agents/media_agents.py` *(Fixes 1, 2, 3 — zero edits under WO#13 itself)*
- `services/ai/keys.py` *(Fix 4)*
- `domains/workout/routers/workout_plan_ai_generator.py` *(Fix 5, on top of its Part A edit)*
- `domains/media/routers/media_recommend.py` *(Fix 5, on top of its Part A edit)*

**Created — reconstructed stubs, not source material (see Known
Caveats):**
- `services/ai/base.py`
- `services/ai/keys.py`
- `services/ai/__init__.py`

**Read only, never edited:**
- `services/tmdb_service.py`

---

## Known Caveats / Unverified Assumptions

**This is the most important section for whoever picks this up next.**

1. **`services/ai/base.py`, `services/ai/keys.py`, `services/ai/__init__.py`
   are reconstructed, not real.** None of these three files were ever
   supplied as source material anywhere in this engagement — their
   existence was only inferable from how other files imported and called
   them (e.g. `services/ai/providers/gemini.py`'s
   `from services.ai.base import post_with_retry`,
   `from services.ai.keys import get_provider_key`;
   `job_agents.py`'s `from services.ai import call_gemini_json, MODEL_FLASH_LITE`).
   Every behavior attributed to them in this document — the
   `post_with_retry` retry/backoff schedule, `get_provider_key`'s
   exception behavior, what `services/ai/__init__.py` exports — is an
   inference, not a confirmed fact. **Before merging any of this work,
   swap in the real files and re-run every payload-shape and
   backward-compatibility check in Part A against them.** If the real
   `post_with_retry` has different retry/timeout semantics than the
   stub, some of the "identical to pre-migration behavior" claims in
   Part A may no longer hold and need re-verification.
2. **No live API calls were made anywhere in this engagement.** Every
   "identical payload" claim is a simulation of the payload-construction
   logic, not an actual request against Gemini or TMDB.
3. **No live FastAPI/SQLAlchemy/database stack was available.** The two
   router-level acceptance criteria in Part A were marked ⚠️ for exactly
   this reason. Actually exercising `POST /workout/plans/generate` and
   `POST /media/recommend/generate` end-to-end has not happened.
4. **`GOVERNANCE.md` and `models.py` were never read in this
   engagement.** Every reference to them in this document and in the
   source files themselves (e.g. "GOVERNANCE.md §2.2", "six known
   duplicate AI-client implementations tracked in GOVERNANCE.md §2.3")
   is taken on faith from what other files say about them. See the next
   section for what needs checking there.

---

## Post-Migration Follow-Up Required

This section is written for whoever picks up work after WO#14, WO#15,
WO#16 (and this postmortem's Part B fixes) have all landed. Treat it as
a checklist, not a narrative.

### 1. Replace the reconstructed `services/ai/` stub files
Before anything else. Swap in the real `base.py`, `keys.py`,
`__init__.py`. Re-run:
- Every payload-shape simulation in Part A (job_agents.py,
  recipe_agents.py, weekly_agents.py, workout_plan_ai_generator.py,
  media_recommend.py, blog_agents.py's 4 call sites) against the real
  `post_with_retry`.
- Fix 4's behavior (`get_provider_key` raising on a missing key) against
  the real `keys.py` — the reconstructed version's fix may be redundant,
  wrong, or already handled differently in the real file.

### 2. `blog_agents.py` — final shell-level cleanup pass
Flagged by WO#13's own "For the next work order" note: once WO#14 (Groq)
and WO#15 (Cerebras) also migrate their respective pieces out of this
file, `blog_agents.py` will contain little beyond the public `agent_*`
function shells and non-provider logic (file-type detection, difficulty
distribution validation, prompt-string construction). At that point:
- Confirm the module docstring's routing table is either updated to
  reflect the new `services/ai/`-backed reality, or explicitly marked as
  historical (WO#16 apparently intends to duplicate it into
  `services/ai/README.md` rather than delete it — confirm that actually
  happened and this file's copy doesn't drift out of sync).
- Check whether the `AGENT N — AGENT NAME` section-header comments and
  "Frequency"/"Provider" annotations throughout the file still make
  sense once every agent function is a thin call into `services/ai/`
  rather than owning its own HTTP logic.

### 3. `GOVERNANCE.md` — not read in this engagement, needs a pass
Referenced constantly (§1.2 router line-count ceiling, §2.2 DAG import
boundary, §2.3 AI Service Layer target state and "six known duplicate
AI-client implementations" tracking list) but never actually opened.
Before closing out this whole migration program:
- Confirm what the original "six known duplicate AI-client
  implementations" actually enumerated. Based on this engagement's
  visibility: `job_agents.py` (WO#11), `recipe_agents.py` (WO#12),
  `weekly_agents.py`, `workout_plan_ai_generator.py`,
  `media_recommend.py`, and `blog_agents.py`'s Gemini functions (all
  WO#13) are now migrated. `blog_agents.py`'s Groq and Cerebras logic
  are explicitly **not** migrated yet (WO#14, WO#15). If the original
  "six" count included Groq/Cerebras as separate entries, the count
  above may not add up to six — reconcile against the real document
  rather than this summary.
- Update whatever tracking mechanism §2.3 uses (checklist, table, etc.)
  to reflect what's actually done as of this postmortem.
- Confirm §2.2's DAG-import-boundary rule is still accurately described
  everywhere it's cited (`media_agents.py`'s docstring, `job_agents.py`'s
  docstring) — none of those citations were checked against the real
  GOVERNANCE.md text in this engagement, only assumed accurate.

### 4. `domains/media/models.py` and `domains/workout/models.py` —
### not read in this engagement, needs a pass
Neither file's actual content was available anywhere in this
engagement — every reference to `MediaItem`, `Exercise`, `WorkoutPlan`,
etc. came from how *other* files imported and used them, never from
reading the model definitions themselves. Specifically worth checking
once available:
- **`MediaItem.media_type`**: Fix 2 in Part B assumes this column/field's
  TV-content value is the literal string `"tv_show"`, inferred from
  `tmdb_service.py::search_tv()` setting `"media_type": "tv_show"` on
  every result it normalizes, and from `media_recommend.py`'s
  `RecommendationMediaType` handling `"movie" | "tv_show" | "book" | "any"`.
  If `MediaItem.media_type` is a SQLAlchemy/Python `Enum` rather than a
  plain string, confirm the enum member name and value match what
  `media_agents.py` now accepts, and confirm there isn't a third
  representation (e.g. `TV_SHOW` vs `"tv_show"`) that would silently
  break the comparison.
- **Any other AI-service or TMDB-service references inside these model
  files** — the requester's original ask for this postmortem
  specifically named "adjust the models.py, removing the references from
  there" as an example. No such references were found or confirmed in
  this engagement because the files were never read. If `models.py` (in
  either domain) contains inline comments, docstrings, or logic referring
  to the old bespoke `_gemini_flash`/`_gemini_flash_json`-style calls, the
  old TMDB duplication as an open problem (now partially resolved per
  Part B), or anything else this migration has since made stale, those
  need the same treatment `media_agents.py`'s docstring got in Fix 3:
  don't delete history, but stop presenting resolved issues as open ones.

### 5. Audit for other async-route-calls-sync-services/ai/ instances
Fix 5 in Part B fixed exactly two call sites — the only two visible in
this engagement's file set. **This was not a codebase-wide audit.** Any
other FastAPI router anywhere in the project that calls into
`services/ai/` from an `async def` handler has the same event-loop-
blocking exposure and has not been checked. Search for `call_gemini_json(`,
`call_gemini_text(`, and `call_gemma_json(` call sites across the full
`domains/` tree (not just `workout/` and `media/`) and confirm each one
either isn't in an async context, or is wrapped the same way
(`asyncio.to_thread`) as the two fixed here.

### 6. Re-diff `media_agents.py` vs `tmdb_service.py` if either changes
Per the living-requirement comment Fix 3 left in `media_agents.py`'s
docstring: the two files still intentionally duplicate logic rather than
share it (GOVERNANCE §2.2). Any future change to either file's
watch-provider logic needs a manual re-diff against the other — there is
no automated check enforcing they stay in sync, by design.

### 7. Optional, low-priority: `tmdb_service.py`'s provider-ID extraction
`media_agents.py` defensively filters out entries missing a
`"provider_id"` key; `tmdb_service.py` doesn't. Not a bug, not required,
but if `tmdb_service.py` is touched for an unrelated reason, consider
bringing it up to the same defensiveness.

### 8. Test suite / CI audit
Not checked in this engagement — no test files were provided or visible.
Before merging, search test suites and CI config for:
- References to the now-deleted `blog_agents._gemini_flash`,
  `blog_agents._gemini_flash_json`, `blog_agents._gemini_key`.
- Any test asserting `call_gemini_json`'s old positional signature
  `(system, prompt, schema, model, retries)` — these would break under
  the new `(prompt, schema, system, model, retries)` signature the same
  way the real `job_agents.py`/`recipe_agents.py` call sites would have,
  and need the same explicit-keyword-argument fix.
- Any test mocking `media_agents.get_tmdb_watch_providers()`'s old
  behavior (raises on 404, rejects `"tv_show"`) that would now need
  updating to match Fixes 1 and 2.

### 9. WO#14 (Groq) and WO#15 (Cerebras) — still outstanding
Not started in this engagement. WO#14's own scope note (per WO#13's
"For the next work order" section) should be read as covering whatever
remains in `blog_agents.py` *after* this postmortem's Part A changes,
not the full original file — the Gemini-shaped functions and their three
helpers are already gone.

---

## Reviewer Sign-Off Checklist

- [ ] Real `services/ai/base.py`, `keys.py`, `__init__.py` swapped in;
      all payload-shape and backward-compat checks re-run against them
- [ ] `GOVERNANCE.md` read and cross-checked against every citation of it
      in the files this engagement touched
- [ ] `domains/media/models.py` and `domains/workout/models.py` read;
      `MediaItem.media_type`'s `"tv_show"` assumption confirmed or
      corrected; any stale references to resolved issues updated
- [ ] Full-codebase audit for other async-calls-sync-`services/ai/`
      instances beyond the two fixed here
- [ ] Test suite / CI checked for references to deleted symbols and old
      signatures
- [ ] WO#14 and WO#15 scoped and run
- [ ] `blog_agents.py` shell-cleanup pass done once WO#14/#15 land
- [ ] This document itself updated to reflect what actually happened in
      each of the above, rather than left as a static snapshot
