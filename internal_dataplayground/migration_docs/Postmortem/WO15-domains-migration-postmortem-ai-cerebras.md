# WO#15 — Cerebras Provider Migration — Postmortem & Post-Migration Requirements

**Status:** ✅ Executed and internally verified against a mocked SDK harness.
**Not yet reviewed against the real repository or a live Cerebras endpoint** —
see "Known Caveats" (§5) before treating anything here as final.

> **Correction notice (post-delivery, two rounds):** the deliverable
> shipped with this postmortem initially contained several
> structurally-equivalent but **trimmed** reconstructions, built as a
> lightweight sandbox test harness and then mistakenly shipped as the
> actual code artifacts instead of the real files. **Round 1** caught
> `blog_agents.py` (docstrings shortened, several functions' full bodies
> missing, `_detect_file_type()` missing branches). **Round 2**, prompted
> by a direct question about whether every other file shipped correctly,
> found the same pattern in three more files that had been labeled
> "included unmodified for context": `services/ai/base.py`,
> `services/ai/providers/gemini.py`, and `services/ai/providers/groq.py`
> all had their module/function docstrings and explanatory comments
> stripped (functional code was intact in these three — only prose was
> missing), and `services/ai/__init__.py` had lost three lines from its
> header comment. **All are now fixed**, verified byte-for-byte identical
> to the original source documents via `diff` (not eyeballed), and the
> full 8-scenario mocked-SDK verification in §4.4 was re-run against the
> fully-restored file set with identical results. The version now
> included in the deliverable has been checked this way file-by-file, not
> just for the one file that was first caught.

**Series:** AI Service Layer, GOVERNANCE.md §2.3. Fourth of six
(WO#11 → WO#12 → WO#13 → WO#14 → **WO#15 (this one)** → WO#16). This is the
**last individual provider migration** in the series — Gemini (WO#11–13),
Groq (WO#14), and Cerebras (WO#15, this document) are now all represented
in `services/ai/providers/`. WO#16 (capstone) is the only work order left
in this track, and this document's §6 is written specifically to hand it
everything it needs.

**Read this document if:** you are reviewing WO#15 for pass/fail, you are
about to scope or execute WO#16, or you are doing final AI Service Layer
cleanup and need a single, current account of what state `blog_agents.py`
and `services/ai/` are actually in.

**How to read this document, per the request that produced it:** §2 is the
original Work Order text, unamended — what was drafted before anyone knew
WO#13/#14 would already have landed by execution time. §3 is the one
change agreed to that text *before* execution (the precondition-check
amendment). §4 is what was actually done, with verification evidence, so
a reviewer can check execution against both §2 and §3 independently. §6 is
the forward-looking requirements list — written to be consumed as a
checklist by whichever agent or engineer picks up WO#16 or a final
cleanup pass, not as narrative.

---

## 1. Executive Summary

WO#15 extracted `blog_agents.py`'s `_cerebras()` — the most complex and
highest-stakes function in the whole AI Service Layer migration series,
given its production-tuned 429/503 retry logic and its
`(content, remaining_tokens)` return contract with
`life_os_code_improve.py`'s DAG — into a new
`services/ai/providers/cerebras.py` module, with zero intended behavior
change. All four callers (`agent_code_narrator`, `agent_refiner`,
`agent_code_commenter`, `agent_code_improver`) were repointed. No live
Cerebras access was available, so verification was performed against a
purpose-built fake `cerebras.cloud.sdk` module that reproduces the SDK's
raw-response and exception-raising code paths under test control — this
let every retry branch actually be *executed*, not just read, both before
and after the migration, and diffed programmatically rather than by
inspection.

**Net result:** 2 files edited (`airflow/agents/blog_agents.py`,
`services/ai/__init__.py`), 1 file created
(`services/ai/providers/cerebras.py`), 1 file confirmed to need no edit
(`services/ai/keys.py` — the `"cerebras"` key mapping was already present
from the WO#11-era stub). All 11 applicable acceptance criteria passed;
one criterion (the "leave the commented-out prior `_cerebras()`
implementation untouched" instruction) had nothing to verify against — no
such block exists in the source this session was given. See §4.5 and §5.

---

## 2. What the *original* Work Order specified

Unamended baseline, preserved here so a reviewer can see exactly what
changed between "drafted" and "executed" without diffing two documents.

- **ROLE:** senior refactoring engineer relocating rate-limit-hardened,
  safety-critical code with zero tolerance for behavioral drift — read the
  function twice before writing anything; stop and report on any
  ambiguity rather than guessing at an "obviously correct" simplification.
- **HARD BOUNDARIES (as originally drafted):**
  - Preserve `_CEREBRAS_BACKOFF = [75, 150, 300, 600]` unchanged; preserve
    both the raw-HTTP-status and SDK-exception handling paths for 429
    (`RateLimitError`) and 503 (`APIStatusError`); preserve the
    `Retry-After` override on 429.
  - Preserve the `(content: str, remaining_tokens: int)` return shape
    exactly — `life_os_code_improve.py`'s DAG depends on it.
  - Preserve both `_CEREBRAS_QWEN3` and `_CEREBRAS_LLAMA33` as separate
    exported constants, even though only Qwen3 appears to be used —
    report the discrepancy, don't silently drop the unused one.
  - Do not "fix" the Refiner model mismatch (docstring says Llama 3.3 70B,
    code actually calls Qwen3) — migrate it exactly as it behaves today,
    report the mismatch as a separate product decision.
  - Only the four callers' `_cerebras(...)` call sites change — no
    surrounding logic in any of the four agent functions.
  - Confirm via grep that `_cerebras_key()` has no caller besides
    `_cerebras()` before deleting it.
- **SCOPE:** new `services/ai/providers/cerebras.py`; edit
  `services/ai/__init__.py` (re-export) and `services/ai/keys.py` (add
  Cerebras key lookup); edit `blog_agents.py` (the named symbols and four
  call sites only) — explicitly **leave the large commented-out prior
  `_cerebras()` implementation block exactly as-is**, noted as a future
  WO#19-style cleanup candidate, not touched here.
- **STEPS:** (1) add `get_provider_key("cerebras")` to `keys.py`;
  (2) create `cerebras.py` with `call_cerebras_text(model, system, prompt,
  temperature=0.3, max_tokens=4096) -> tuple[str, int]`, extracting
  `_cerebras()` verbatim, importing the Cerebras SDK classes lazily inside
  the function body exactly as the original did; (3) export
  `call_cerebras_text`, `MODEL_QWEN3`, `MODEL_LLAMA33` from
  `services/ai/__init__.py`; (4) delete the live `_cerebras()` (not any
  commented-out version), `_cerebras_key()`, `_CEREBRAS_BACKOFF`,
  `_CEREBRAS_QWEN3`, `_CEREBRAS_LLAMA33` from `blog_agents.py` after
  confirming via grep nothing else depends on them; repoint all four call
  sites to `call_cerebras_text(MODEL_QWEN3, ...)`, keeping Refiner on
  Qwen3.
- **ACCEPTANCE CRITERIA (as originally drafted):** identical HTTP/SDK
  request shape; correct behavior on all four retry branches (429+header,
  429 no header, 503, exhausted); correct 2-tuple return shape on every
  success path; the three content-only callers correctly discard the
  token count; `agent_code_improver` still returns the full tuple and the
  DAG contract still holds; the five old symbols confirmed gone via grep,
  with the commented-out block confirmed **still present and untouched**;
  Refiner confirmed still using Qwen3; a final word-boundary grep sweep.

At the time this WO was originally drafted, "every other function...
remains completely untouched" implicitly meant "unchanged relative to the
file as it existed before *any* AI Service Layer migration." That
assumption was already known to be stale by the time WO#14 ran (see its
own postmortem, §3) — the same correction applies here, one layer deeper.

---

## 3. What was agreed to change *before execution* (the amendment)

The amendment supplied alongside this WO added exactly one paragraph to
HARD BOUNDARIES — no change to STEPS or ACCEPTANCE CRITERIA content:

> **By the time this WO executes, `blog_agents.py` will already have had
> its Gemini functions (WO#13) and its Groq function (WO#14) removed.**
> The file you're diffing "everything else is unchanged" against is
> smaller than the version this WO was originally drafted against —
> confirm you're starting from a state where `_gemini_flash` and
> `_groq_llama` are both already gone before proceeding.

**Why this matters for review:** the "completely untouched" criterion for
every non-Cerebras function in `blog_agents.py` can only be checked
against the **WO#14 post-migration baseline** of the file (import line
already reading `from services.ai import MODEL_FLASH, call_gemini_json,
call_gemini_text, call_groq_text`, with no private Gemini or Groq helpers
present) — not against any earlier snapshot. Diffing against a version
that still contains `_gemini_flash` or `_groq_llama` would produce
false-positive "unexplained changes" findings.

**Precondition check performed before execution, per the amendment's own
instruction:** confirmed the input file's import line already read
`from services.ai import MODEL_FLASH, call_gemini_json, call_gemini_text,
call_groq_text` and that no `_gemini_flash`, `_gemini_flash_json`,
`_gemini_key`, `_groq_llama`, or `_groq_key` definitions existed anywhere
in the file. WO#13 and WO#14 had both landed. Execution proceeded.

---

## 4. What was actually executed

### 4.1 Files created

| File | Purpose |
|---|---|
| `services/ai/providers/cerebras.py` | `call_cerebras_text(model, system, prompt, temperature=0.3, max_tokens=4096) -> tuple[str, int]`, `MODEL_QWEN3`, `MODEL_LLAMA33`, `_CEREBRAS_BACKOFF`. Verbatim extraction — same dual-path 429/503 handling, same `Retry-After` override, same token-remaining parsing with its `(ValueError, TypeError)` fallback to `0`, same exhausted-retries `RuntimeError` with `last_exc` detail. Cerebras SDK classes (`Cerebras`, `APIStatusError`, `RateLimitError`) still imported lazily inside the function body, matching the original's deliberate pattern. |

### 4.2 Files moved

None.

### 4.3 Files edited

| File | Change |
|---|---|
| `airflow/agents/blog_agents.py` | Import line: added `call_cerebras_text, MODEL_QWEN3`. Deleted `_cerebras_key()`, `_cerebras()`, `_CEREBRAS_BACKOFF`, the `# ── CEREBRAS MODEL IDs ──` header, `_CEREBRAS_QWEN3`, `_CEREBRAS_LLAMA33`. Updated all four call sites to `call_cerebras_text(MODEL_QWEN3, ...)`, same arguments, same order. `agent_refiner` still calls with `MODEL_QWEN3` — not corrected to Llama 3.3 (§4.6, finding 2). Diff is isolated to exactly these hunks — every other function (`agent_readme_writer`, `agent_ghostwriter`, `agent_editor`, `agent_idea_expander`, `_detect_file_type`, `_estimate_tokens`, both JSON schemas, difficulty/type validation helpers, module docstring) is byte-identical to the WO#14 end-state. |
| `services/ai/__init__.py` | Added a `# ── Added by WO#15 ──` import block re-exporting `MODEL_QWEN3`, `MODEL_LLAMA33`, `call_cerebras_text`; extended `__all__` to match, following the exact one-block-per-provider convention WO#13 (Gemini) and WO#14 (Groq) already established. |
| `services/ai/keys.py` | **No edit made.** `get_provider_key("cerebras")` → `CEREBRAS_API` was already present in `_ENV_VAR_BY_PROVIDER` — confirmed by direct inspection before treating Step 1 as satisfied, not assumed. Same "already present, nothing to do" outcome WO#14 found for `"groq"`. |

Full diff for the two edited files is included alongside this document at
`migration_docs/verification/wo15_cerebras/blog_agents_and_init.diff`.

### 4.4 Verification performed

No live Cerebras API access was available in this environment, and the
`cerebras-cloud-sdk` PyPI package (pinned unversioned in
`Dockerfile.airflow`, not installed in this session) was not available
either. Rather than verifying by static inspection alone, a **controllable
fake `cerebras.cloud.sdk` module** was built — reproducing
`Cerebras(...).with_raw_response.chat.completions.create(...)`'s raw-status-code
response shape, plus `APIStatusError`/`RateLimitError` exception classes —
so that every retry branch could actually be *executed under test*, not
just read.

**Method (four-artifact pattern, per WO#11 postmortem §1.3, reused
throughout this series):**
1. **Baseline capture** — ran the real pre-migration `_cerebras()` against
   the fake SDK across 8 scenarios (see below), recording exact request
   kwargs, exact `time.sleep()` call arguments, exact return values, and
   exact exception messages.
2. **Migrated capture** — ran the new `call_cerebras_text()` (both
   directly and via all four real `blog_agents.py` callers) through the
   identical scenarios against the identical fake SDK.
3. **Programmatic diff** — compared the two captures as data, not by eye.
4. **Repo-wide grep sweep** — confirmed the five deleted symbols have zero
   live references anywhere in `blog_agents.py`.

**Scenarios exercised (both pre- and post-migration):**

| # | Scenario | Confirms |
|---|---|---|
| 1 | Success, 200 | Request shape: model, `messages` (system+user roles), `temperature`, `max_tokens` |
| 2 | 429 with `Retry-After` header | Sleeps for the header's exact value (`12.0`s), not the backoff schedule |
| 3 | 429, no header (raw-response path) | Falls back to `_CEREBRAS_BACKOFF[attempt]` (`75`s) |
| 4 | 503 (raw-response path) | Uses backoff schedule (`75`s), not `Retry-After` logic |
| 5 | `RateLimitError` (SDK exception path) | Uses the exception's `response.headers["retry-after"]` (`9`s) |
| 6 | `APIStatusError`, status 503 (SDK exception path) | Retries using backoff schedule (`75`s) |
| 7 | `APIStatusError`, status 400 (non-retriable) | Raises immediately, zero sleeps |
| 8 | All 4 backoff attempts exhausted (429s) | `RuntimeError` with exact message format including `last_exc`; sleep sequence `[75, 150, 300, 600]` |

**Result: all 8 scenarios byte-identical between baseline and migrated
captures** (verified programmatically, not eyeballed — see
`migration_docs/verification/wo15_cerebras/` for the raw JSON captures and
the scripts that produced them).

**Additional runtime checks performed** (not just request-shape diffing):
- All four real `blog_agents.py` callers invoked end-to-end against the
  fake SDK: `agent_code_narrator`, `agent_refiner`, `agent_code_commenter`
  each confirmed to return a plain `str` (second tuple element correctly
  discarded); `agent_code_improver` confirmed to return the full
  `(str, int)` tuple.
- `agent_refiner`'s actual outbound `model` kwarg confirmed at runtime to
  be `"qwen-3-235b-a22b-instruct-2507"`, not the Llama 3.3 ID — matching
  the pre-existing behavior HARD BOUNDARIES required preserving.
- **DAG contract simulation:** called `agent_code_improver()` with a fake
  200 response carrying `x-ratelimit-remaining-tokens-minute: 2500`,
  confirmed the returned `remaining_tokens` is a plain `int` equal to
  `2500`, then ran `life_os_code_improve.py`'s own decision logic
  (`remaining_tokens < 3000` → sleep) against that value and confirmed it
  evaluates correctly (`True`, i.e. the DAG would sleep before the next
  file — the correct behavior for a nearly-exhausted token budget).
- `ast.parse()` on every created/edited file — all parse cleanly.
- Repo-wide grep for the five deleted symbol names — zero hits in code
  outside the new `services/ai/providers/cerebras.py` (where they appear
  only in docstring prose describing what was moved, plus `cerebras.py`'s
  own re-declared `_CEREBRAS_BACKOFF`, which is a different, local
  constant now owned by the provider module, not the deleted one).

### 4.5 A note on the "commented-out prior `_cerebras()` block" instruction

Both the original WO text and the amendment assume this block exists and
instruct leaving it untouched. **It does not appear anywhere in the
`blog_agents.py` content this session was given** — the file goes
directly from the "PROVIDER CALL HELPERS" section header into the live
`_cerebras()` function and its constants, with nothing commented out above
it. This was checked explicitly (`grep "# def _cerebras"`) before editing,
not assumed absent. Two possibilities, neither resolvable from here: the
block was already removed by an untracked prior cleanup pass, or this
session's copy of the file simply doesn't include it. WO#14's own
postmortem (§9.8) flagged this exact same uncertainty and could not
resolve it either — **this is now the second consecutive work order in
this series unable to confirm the block's existence.** Whoever has the
real repository should settle this directly before WO#19 (dead-code
removal) runs, since WO#19's own instructions locate this block using
landmarks (the live `_cerebras()` function, the `# ── CEREBRAS MODEL IDs
──` header) that this WO has now removed — see the amendments file's own
fallback instruction for WO#19 if the block turns out to exist.

### 4.6 Acceptance criteria — final determination

| # | Criterion | Result | Basis |
|---|---|---|---|
| 1 | Identical HTTP/SDK request shape | ✅ | Scenario 1, programmatic diff match |
| 2 | 429 + `Retry-After` — sleeps header value, not schedule | ✅ | Scenario 2, match (`12.0`s both) |
| 3 | 429, no header — falls back to schedule | ✅ | Scenario 3, match (`75`s both) |
| 4 | 503 raw path — uses schedule, not `Retry-After` logic | ✅ | Scenario 4, match (`75`s both) |
| 5 | Exhausted retries — `RuntimeError`, same message + `last_exc` | ✅ | Scenario 8, message and sleep sequence identical |
| 6 | Return shape: 2-tuple `(str, int)` on every success path | ✅ | Confirmed on direct call + all 4 real callers |
| 7 | Narrator/Refiner/Commenter discard 2nd element, return `str` | ✅ | Runtime type check on all three |
| 8 | `agent_code_improver` returns full tuple; DAG contract holds | ✅ | Runtime tuple check + simulated DAG decision logic |
| 9 | Five old symbols confirmed gone (grep) | ✅ | Zero hits in `blog_agents.py` |
| 9b | Commented-out prior `_cerebras()` block left untouched | ⚠️ | No such block exists in the provided source — see §4.5. Nothing to verify against; nothing was deleted. |
| 10 | Refiner still calls with Qwen3 | ✅ | Confirmed statically and at runtime |
| 11 | Final word-boundary grep sweep | ✅ | Zero live-code hits |

**11/11 applicable criteria ✅. 1 criterion (9b) marked ⚠️ — not a failure
of the migration, but an unresolved fact about the source material that
predates this WO and that this session cannot settle unilaterally.**

---

## 5. Known Caveats / Unverified Assumptions

**Read this section before merging anything from this WO.**

1. **`services/ai/base.py`, `services/ai/keys.py`, `services/ai/__init__.py`
   are still reconstructed stubs**, as they have been since WO#11. No file
   in this three-file set has ever been supplied as real source material
   anywhere across WO#11 through WO#15 — every behavior attributed to them
   (in particular `services.ai.keys.get_provider_key`'s exception-vs-`None`
   behavior on a missing key, which this WO's Cerebras key lookup now also
   depends on by inheritance) is an inference from call-site usage, not a
   confirmed fact. **This caveat compounds with every provider WO that
   edits these files** — WO#13 (Gemini), WO#14 (Groq), and now WO#15
   (Cerebras) have all added exports to the same reconstructed
   `__init__.py`. Before any of this is merged, swap in the real files and
   re-run every payload-shape and retry-path check in this document (and
   WO#13's, WO#14's) against them — see §6.1.
2. **No live Cerebras API call was made.** Every retry-path and
   request-shape claim in §4.4 is against a hand-built fake SDK module
   whose shape (`Cerebras(...).with_raw_response.chat.completions.create()`
   returning an object with `.status_code`, `.headers`, `.json()`; separate
   `APIStatusError`/`RateLimitError` exception classes) was inferred from
   reading the pre-migration `_cerebras()` code, not from the installed
   `cerebras-cloud-sdk` package (which was not installed in this
   environment and is pinned **unversioned** in `Dockerfile.airflow` — a
   future SDK version could change this shape without anyone in this
   engagement noticing). Recommend a live smoke test, or at minimum a test
   against the actual installed SDK version, before this is considered
   fully verified.
3. **`GOVERNANCE.md` and both domain `models.py` files
   (`domains/media/models.py`, `domains/workout/models.py`) were not read
   in this session** beyond what was directly provided. Every reference to
   them elsewhere in this document is inherited from earlier postmortems
   in this chain, not independently re-verified here.
4. **The root `models.py`** (document provided in this session) **was
   read directly** and is addressed explicitly in §6.4 below — this one
   caveat is resolved, not open.

---

## 6. Post-Migration Follow-Up Required

**This section is written to be read as a checklist by whichever agent or
engineer executes WO#16 or a final AI Service Layer cleanup pass — not as
a narrative.** It consolidates open items carried forward from the WO#11,
WO#13, and WO#14 postmortems with what's newly known or newly actionable
now that WO#15 (the last individual provider migration) is done. Where an
item is inherited rather than newly discovered here, its origin is noted
so nothing gets silently re-attributed. Do not consider the AI Service
Layer initiative closed until every item below is either done or
explicitly re-deferred with a stated reason, mirroring the discipline
GOVERNANCE.md §4.6 applies to domain migrations.

### 6.1 Replace the three reconstructed `services/ai/` stub files — highest priority, blocks trusting anything else

Before anything else in this list. `services/ai/base.py`,
`services/ai/keys.py`, `services/ai/__init__.py` have been reconstructed,
never verified, since WO#11 (see §5, item 1). Once the real files are
available:
- Re-run every payload-shape simulation across the **entire** series
  against the real `post_with_retry` and `get_provider_key`:
  `job_agents.py` (WO#11), `recipe_agents.py`'s four call sites (WO#12),
  `weekly_agents.py`, `workout_plan_ai_generator.py`,
  `media_recommend.py`, `blog_agents.py`'s Gemini call sites (all WO#13),
  `blog_agents.py`'s Groq call site (WO#14), and this WO's four Cerebras
  call sites.
- Specifically re-run this WO's 8 retry-path scenarios (§4.4) against the
  **real** `cerebras-cloud-sdk` package (or a fake built directly against
  its actual installed version, not inferred from application code) —
  the stub caveat in §5 item 2 means the SDK-shape assumptions here are
  the least-verified claim in this whole postmortem.
- Confirm `services/ai/keys.py`'s exception-on-missing-key behavior is
  actually what the real file does — if the real file silently returns
  `None` instead, every provider's "fails loud" assumption inherited from
  this reconstruction needs re-examination.

### 6.2 `blog_agents.py` — final shell-level cleanup pass (now unblocked)

With Gemini (WO#13), Groq (WO#14), and Cerebras (WO#15, this document) all
extracted, `blog_agents.py` should now contain **zero** inline
provider-calling implementations. Verify:
- [ ] `grep -n "^def _"` returns only `_detect_file_type` and the
  (duplicate — see next bullet) `_estimate_tokens` definitions. No
  `_gemini_*`, `_groq_*`, or `_cerebras*` helper remains.
- [ ] Pre-existing, not this WO's to fix: `_estimate_tokens()` is defined
  twice in the file (flagged in WO#14's postmortem §9.4, confirmed still
  present in this session's copy of the file). Report per GOVERNANCE §4.5,
  don't silently collapse it into a cleanup WO's diff without flagging it
  first.
- [ ] **New findings from this WO, all currently left in place per the
  "don't bundle cleanup into a migration diff" discipline established in
  WO#14's postmortem — resolve as their own ticket:**
  - `import os` — now dead. It was used only by the just-deleted
    `_cerebras_key()`.
  - `import time` — was already effectively dead before this WO (only
    exercised via a local re-import shadowing it inside the now-deleted
    `_cerebras()`); still dead.
  - `import requests` — confirmed still dead, unchanged status from
    WO#14's own finding (§7, item 2 of that postmortem).
  - The `# ── KEY HELPERS ──` section header is now empty.
  - `_CEREBRAS_INTER_REQUEST_SLEEP = 65` is an orphaned constant with no
    in-file consumer — it was *not* in this WO's explicit deletion list
    (Step 4 named only `_cerebras_key`, `_cerebras`, `_CEREBRAS_BACKOFF`,
    `_CEREBRAS_QWEN3`, `_CEREBRAS_LLAMA33`), so it was left in place
    rather than guessed at. Confirm with whoever owns the real file
    whether it has any purpose beyond documentation before removing it —
    the real sleep timing for `life_os_code_improve.py` lives in that
    DAG's own separate `INTER_REQUEST_DELAY_SEC` constant, so this one
    may simply be vestigial.
- [ ] The import line should now read something like `from services.ai
  import MODEL_FLASH, call_gemini_json, call_gemini_text, call_groq_text,
  call_cerebras_text, MODEL_QWEN3` — confirm this matches whatever the
  real file's final Step-5-of-WO#13-through-Step-4-of-WO#15 accumulated
  state is.
- [ ] Confirm the commented-out prior `_cerebras()` implementation block
  either exists (in which case WO#19's original landmark-based
  instructions for finding it are now broken — see §4.5 above and the
  amendments file's own fallback for WO#19) or genuinely doesn't (in which
  case that whole WO#19 sub-task is moot and should be marked as such,
  not silently skipped).
- [ ] The module docstring's MODEL ROUTING table and ROUTING RATIONALE
  section — per GOVERNANCE.md §2.3's target state, this belongs in
  `services/ai/README.md`, "moved from blog_agents.py's header comment."
  As of this postmortem, **nothing in WO#13, #14, or #15 has performed
  this relocation** (flagged as unowned back in WO#14's postmortem §9.7,
  still unowned now). Recommend folding it into WO#16 explicitly, since
  WO#16 is the natural last stop in this series.

### 6.3 Design and build the provider-agnostic `call_ai_text()` / `call_ai_json()` wrapper — now unblocked

This was deliberately deferred at WO#11, WO#12, and WO#13 pending enough
real provider shapes to generalize from safely. **All three now exist**
(Gemini: schema-enforced JSON, native `systemInstruction`; Groq: OpenAI-style
messages array, no retry logic at all; Cerebras: `(content,
remaining_tokens)` tuple return, `Retry-After`-aware backoff distinct from
Gemini's). WO#16 should resolve, not re-defer, the open questions already
on record (WO#11 postmortem §3.3, restated here since they're now
answerable):
- Does the wrapper always return just `content`, dropping Cerebras's
  token-budget signal (breaking `life_os_code_improve.py`'s use of it
  unless that DAG is also updated — which is out of scope for a
  service-layer WO per the standing DAG/FastAPI boundary), or does it
  return a richer, uniform object across all three providers even where
  Gemini/Groq have no equivalent field?
- Does the wrapper add retry uniformly (a real behavior change for Groq,
  which currently has none) or leave retry behavior provider-specific?
- Does `services/ai/base.py::post_with_retry` get extended to also cover
  Cerebras's `Retry-After`-aware, differently-scheduled retry logic, or
  does Cerebras keep its own separate retry implementation permanently
  because a shared one doesn't actually serve both providers well?
  (This WO's own extraction did **not** attempt to route Cerebras through
  `post_with_retry` — it kept the SDK-based retry loop entirely
  self-contained in `services/ai/providers/cerebras.py`, per HARD
  BOUNDARIES' instruction to relocate, not redesign. That decision is
  still open for WO#16 to make deliberately.)

### 6.4 The `models.py` question — addressed directly

**Checked in this session, not inherited:** the root `models.py` provided
in this engagement is the **post-WO#20** state — every domain's re-export
shim has already been removed, and the file is now a documentation stub
holding a set of now-unused imports (`datetime`, `enum`, `math`, SQLAlchemy
column types, `Mapped`/`mapped_column`/`relationship`, `BaseModel`)
plus a comment explaining why they weren't cleaned up (WO#20's own scope
discipline — "removing a shim and updating exactly one import line in
dashboard.py per domain," not general dead-code cleanup).

**Direct finding: this file contains zero references to Cerebras, Gemini,
Groq, any AI provider, any API key, or `services.ai`, and nothing in this
WO's own scope touches it.** This is expected, not a gap — as WO#14's
postmortem (§8) already established explicitly, the AI Service Layer track
(WO#11–16, including this WO) and the Domain Migration track (WO#1–10,
#20, whose shim-removal mechanism is what touches `models.py`) are
**unrelated subsystems that happen to be mid-migration in the same
repository at the same time.** `blog_agents.py` is not an ORM model file,
does not import from `models.py`, and is not consumed by `dashboard.py` or
any other cross-domain reader. There is nothing for this WO, or for WO#16,
to remove from `models.py` as a consequence of the AI Service Layer work.

**What still needs checking, and by whom:** the `models.py` snapshot this
session worked from could be stale relative to the live repository by the
time WO#16 runs. If a future session is specifically asked to "adjust
`models.py`, removing AI-service references from there" (the kind of
instruction that motivated this section), the correct response is:
re-fetch the real, current `models.py` and confirm the same
zero-references finding directly, rather than trusting this postmortem's
snapshot — and if references *are* found, they almost certainly indicate
either (a) someone added inline AI-calling logic directly into a model
file, which is itself a GOVERNANCE.md §2.1 violation worth flagging on its
own, or (b) this document's "unrelated subsystems" premise has stopped
holding for some reason, which would be a significant enough finding to
stop and report rather than quietly fix.

**Separately, and not resolved in this session (inherited from WO#13's
postmortem, unrelated to Cerebras but part of the same broader "what's
left" picture):** `domains/media/models.py` and `domains/workout/models.py`
have never been read in this engagement. WO#13 Part B's Fix 2
(`media_agents.py` accepting `"tv_show"` as a `media_type` value) rests on
an inference about `MediaItem.media_type`'s actual representation that has
never been confirmed against the real model file. This has nothing to do
with Cerebras specifically, but is flagged here again because it remains
open and this document is meant to be a consolidation point.

### 6.5 `finance_upload.py` — the one remaining unmigrated duplicate

With Gemini, Groq, and Cerebras all extracted, `finance_upload.py`'s
`google-genai`-SDK-based call is now the **only** originally-tracked
duplicate AI-client implementation left untouched. WO#11's postmortem
flagged this as architecturally different from everything else in the
series (SDK-based, not raw REST) and recommended a design discussion
before scoping a migration WO, rather than assuming the same
extract-verbatim pattern applies. That discussion still hasn't happened.
WO#16 was already recommended (by WO#15's own "For the next work order"
note, and independently by WO#12's and WO#13's closing notes) as the place
to finally have it — decide whether this becomes a fourth raw-REST-style
`services/ai/providers/` module, or stays SDK-based as a documented,
intentional exception (in which case `services/ai/providers/gemini.py`
needs a second, SDK-based call shape, not just its current REST one).

### 6.6 Vision support — still not designed

`recipe_agents.py::agent_extract_recipe_from_image()`'s vision call
(flagged since WO#12, partially addressed by that WO's Amendment 6 — see
its postmortem — which gave it retry coverage via `post_with_retry`
directly, but never gave it a real service-layer function of its own)
remains unmigrated. Real design work, not a mechanical port — same status
as every prior postmortem in this series has recorded it.

### 6.7 Dependency audit — `requirements.txt` / `Dockerfile.airflow`

- `cerebras-cloud-sdk` (in `Dockerfile.airflow` only, **unversioned** —
  see §5 item 2's caveat about this) — confirmed still genuinely used,
  now by `services/ai/providers/cerebras.py` instead of
  `blog_agents.py` directly. No change to its dependency status, but
  recommend pinning a version now that its exact call shape matters to a
  shared module three other providers' worth of code sits alongside —
  an unannounced SDK update changing `.with_raw_response`'s shape would
  silently break Cerebras-backed agents.
- No Groq SDK — confirmed unchanged, still pure `requests`.
- `google-genai==1.66.0` — fate still undecided, blocked on §6.5.

### 6.8 Audit for other async-route-calls-sync-`services/ai/` instances

Unchanged from WO#13 postmortem's own open item — only two call sites
(`workout_plan_ai_generator.py`, `media_recommend.py`) were ever fixed
with `asyncio.to_thread(...)`, and only because they were the two visible
in that engagement's file set. This was never a codebase-wide audit. Any
FastAPI router anywhere in `domains/` that calls into `services/ai/`
(including, now, `call_cerebras_text`, if any future router ever needs a
Cerebras call synchronously) from an `async def` handler has the same
event-loop-blocking exposure and has not been checked.

### 6.9 `GOVERNANCE.md` §2.3 update

Still not read directly by anyone in this chain (see §5 item 3). Once
WO#16 lands, update it to reflect the now-complete provider set
(`services/ai/providers/{gemini,groq,cerebras}.py`) and resolve the
five-vs-six-duplicates discrepancy WO#11's postmortem first flagged
(§1.5 of that document) — a plausible reconciliation, now that the series
is nearly done, is: `job_agents.py`, `recipe_agents.py`,
`weekly_agents.py`+`workout_plan_ai_generator.py`+`media_recommend.py` (one
bucket, migrated together in WO#13), `blog_agents.py`'s three providers (one
file, three migrations), and `finance_upload.py` — but this is still an
inference pending someone actually reading the governance document's
original §2.3 language.

### 6.10 Test suite / CI audit

Not checked in any session in this chain — no test files have ever been
provided. Before merging this WO's changes, search test suites and CI
config for:
- References to the now-deleted `blog_agents._cerebras`,
  `blog_agents._cerebras_key`, `blog_agents._CEREBRAS_BACKOFF`,
  `blog_agents._CEREBRAS_QWEN3`, `blog_agents._CEREBRAS_LLAMA33`.
- Any test mocking `_cerebras()`'s old positional signature
  (`model, system, prompt, temperature, max_tokens`) — `call_cerebras_text`
  keeps the same positional order, so this is lower-risk than WO#13's
  Gemini signature change, but still worth confirming.
- Any test asserting on `agent_code_improver`'s return type against its
  docstring's (inaccurate) `-> str` annotation rather than its actual
  tuple return — this inaccuracy predates every AI Service Layer WO and
  was intentionally not fixed here (see HARD BOUNDARIES' own instruction
  not to touch it).

### 6.11 Definition of "done" for the full WO#11 → WO#15 provider-extraction chain

Adapting GOVERNANCE.md §4.6's domain-migration definition, and WO#14
postmortem §9.9's version of this same checklist one step further:

- [ ] `blog_agents.py` contains zero inline provider-calling
  implementations (§6.2).
- [ ] `services/ai/providers/` contains `gemini.py`, `groq.py`,
  `cerebras.py` — all present as of this postmortem — each independently
  correct relative to its pre-migration source, no unrequested behavior
  changes.
- [ ] `services/ai/__init__.py` re-exports every provider's public
  functions and model-id constants with no stub/real-file drift remaining
  (§6.1).
- [ ] Every DAG that calls into `blog_agents.py`'s `agent_*` functions
  required zero changes, because public function signatures never
  changed — confirmed for `agent_ghostwriter` (WO#14) and, via this WO's
  DAG-contract simulation, for `agent_code_improver`'s `remaining_tokens`
  contract with `life_os_code_improve.py`. Still owed: the equivalent
  explicit confirmation for `agent_code_narrator`
  (`life_os_code_narrate.py`), `agent_code_commenter`
  (`life_os_code_comment.py`), and `agent_refiner`
  (`life_os_blog_finalizer.py`) — none of these three DAGs depend on the
  tuple return (they all discard the second element at the
  `blog_agents.py` layer already), so they need zero changes regardless,
  but that "needs zero changes" claim has not been independently verified
  against each DAG file's actual import statements the way WO#11 did for
  `job_agents.py`'s consumers.
- [ ] The MODEL ROUTING documentation has an owner and a landing place
  (§6.2, still unowned as of this postmortem).
- [ ] No unrelated behavior changed — dead-code findings (§6.2) and the
  pre-existing duplicate `_estimate_tokens()` are reported, not silently
  fixed, in every WO's diff along the way. This WO complies.

It is *not* considered done if WO#16's diff quietly absorbs any of the
cleanup items in §6.2 without calling them out first — consistent with
every prior WO in this chain holding itself to that same standard.

---

## 7. Rollback reference

`git checkout` on `services/ai/providers/cerebras.py` (delete — it's new),
`services/ai/__init__.py`, and `airflow/agents/blog_agents.py`. No other
files were touched. `services/ai/keys.py` needs no rollback action since
this WO made no edit to it.

---

## 8. Appendix — fingerprints for future automated diffing

```
# Should return nothing (confirms this WO's own extraction is intact):
grep -n "_cerebras_key\|_CEREBRAS_BACKOFF\|_CEREBRAS_QWEN3\|_CEREBRAS_LLAMA33" airflow/agents/blog_agents.py
grep -n "_cerebras\b" airflow/agents/blog_agents.py

# Should return exactly one line, the WO#15-updated import:
grep -n "^from services.ai import" airflow/agents/blog_agents.py

# Should return exactly four lines, one per migrated caller:
grep -n "call_cerebras_text(MODEL_QWEN3" airflow/agents/blog_agents.py

# Should list gemini.py, groq.py, cerebras.py — the full provider set:
ls services/ai/providers/

# Should find nothing — if this returns hits, WO#16 touched blog_agents.py
# in violation of its own HARD BOUNDARIES (per the amendments file's
# correction to WO#16's exclusion list):
git diff <WO15-commit>..<WO16-commit> -- airflow/agents/blog_agents.py
```

Verification scripts, the fake `cerebras.cloud.sdk` stub used to produce
them, and the raw baseline/migrated JSON captures referenced in §4.4 are
included alongside this document under
`migration_docs/verification/wo15_cerebras/`.
