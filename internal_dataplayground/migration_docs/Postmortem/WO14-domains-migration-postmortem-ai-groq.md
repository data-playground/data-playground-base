# Postmortem — Work Order #14: Groq Provider + Ghostwriter Migration

**Status:** ✅ Executed and internally verified. **Not yet reviewed** (see
§6, Reviewer Checklist — this document is written to make that review
possible without re-deriving context).

**Series:** AI Service Layer, GOVERNANCE.md §2.3. Third of six
(WO#11 → WO#12 → **WO#13** → **WO#14 (this one)** → WO#15 → WO#16). This
track is independent of, and shares no files with, the Domain Migration
track (WO#1–10, #20) or the `models.py` shim-removal mechanism — see §8
for why that distinction matters here.

**Read this document if:** you are reviewing WO#14 for pass/fail, you are
about to execute WO#15 or WO#16, or you are doing the eventual AI Service
Layer capstone cleanup and need to know what state `blog_agents.py` and
`services/ai/` were left in and why.

---

## 1. One-paragraph summary

WO#14 extracted `blog_agents.py`'s `_groq_llama()` (and its private key
helper `_groq_key()`) into a new `services/ai/providers/groq.py` module,
re-exported `call_groq_text` / `MODEL_LLAMA_70B` from `services/ai/__init__.py`,
and repointed `agent_ghostwriter()`'s one call site. No other function in
`blog_agents.py` changed. This was the second of three provider
extractions from that file (Gemini in WO#13, Groq here, Cerebras still
pending in WO#15) and leaves `blog_agents.py` with exactly one remaining
inline provider implementation: `_cerebras()`.

---

## 2. What the *original* Work Order specified

This section is the unamended baseline — what WO#14's source document
said before the precondition note was added. Preserved here verbatim in
substance so the reviewer can see exactly what changed between "drafted"
and "executed" (§3) without needing to diff two separate files.

- **ROLE:** senior refactoring engineer, single-provider extraction only.
  Explicit instruction to *not* build a unified Gemini/Groq abstraction in
  this pass.
- **HARD BOUNDARIES (as originally drafted):**
  - Only `_groq_llama()` and its one caller (`agent_ghostwriter()`) move.
  - `_gemini_flash()`, `_gemini_flash_json()`, `_cerebras()`, and every
    other function in `blog_agents.py` remain completely untouched.
  - Do not create `services/ai/providers/cerebras.py` (that's WO#15).
  - No behavior changes: same model, same temperature/max_tokens
    defaults, same endpoint, same header structure.
  - `_groq_key()` is used only by `_groq_llama()` — delete it too, after
    confirming via grep that nothing else in the file calls it.
- **SCOPE:** `services/ai/providers/groq.py` (new), `services/ai/__init__.py`
  (edit), `airflow/agents/blog_agents.py` (edit — `_groq_llama`,
  `_groq_key`, `agent_ghostwriter` only), `services/ai/keys.py` (edit —
  add Groq's key lookup, don't restructure).
- **STEPS:** (1) add `get_provider_key("groq")` to `keys.py`; (2) create
  `groq.py` with `call_groq_text()`, extracting `_groq_llama()` verbatim,
  explicitly **not** adding retry/backoff logic that wasn't already
  present; (3) export from `services/ai/__init__.py`; (4) delete
  `_groq_llama`/`_groq_key` from `blog_agents.py`, repoint
  `agent_ghostwriter()`'s call site.
- **ACCEPTANCE CRITERIA (as originally drafted):** identical HTTP request
  shape (verify via mocked HTTP layer or state the substitution and mark
  ⚠️); `agent_ghostwriter()`'s prompt/system construction unchanged;
  `_groq_llama`/`_groq_key` gone from `blog_agents.py`; every other
  function in the file byte-identical; the calling DAG
  (`life_os_blog_creator.py`) needs zero changes; a repo-wide grep for the
  old names returns zero hits outside the new provider module.

At the time this WO was originally drafted, the assumption baked into
"every other function... completely unchanged" was implicitly "unchanged
relative to the file as it existed pre-any-AI-migration." That assumption
turned out to be wrong by the time this WO actually ran — see §3.

---

## 3. What was agreed to change *before execution* (the amendment)

`work_order_14-20_amendments.md`, written after WO#13 was rewritten to
also remove `blog_agents.py`'s Gemini-shaped functions (`_gemini_flash`,
`_gemini_flash_json`, `_gemini_key`), added exactly one addition to WO#14
— **no change to STEPS or ACCEPTANCE CRITERIA content itself.** The
addition was a new paragraph in HARD BOUNDARIES, inserted after the
existing "every other function... remains completely untouched" line:

> **By the time this WO executes, `blog_agents.py` will already have had
> its Gemini-shaped functions and helpers (`_gemini_flash`,
> `_gemini_flash_json`, `_gemini_key`) removed by WO#13** — that WO now
> owns this extraction, ahead of this one. "Every other function... is
> completely untouched" in this WO's acceptance criteria means untouched
> relative to *WO#13's post-migration state* of the file, not the
> original pre-any-AI-migration state. Confirm this is the state you're
> starting from before diffing — if `_gemini_flash` still exists in the
> file when you start, WO#13 hasn't actually landed yet; stop and report
> rather than proceeding out of order.

**Why this matters for review:** the "completely unchanged" acceptance
criterion cannot be checked against the original document's version of
`blog_agents.py` (document index 14 in the source material, which — note
for the reviewer — is itself already WO#13's post-migration output, not
the pre-WO#13 original; it already shows `_gemini_flash`/`_gemini_flash_json`
absent and the WO#13-shaped import line `from services.ai import
MODEL_FLASH, call_gemini_json, call_gemini_text`). This is the correct
baseline. Diffing against any earlier snapshot of `blog_agents.py` that
still contains `_gemini_flash` would produce false-positive "unexplained
changes" findings.

**Precondition check performed before execution (per the amendment's own
instruction):** confirmed the input file's import line was already
`from services.ai import MODEL_FLASH, call_gemini_json, call_gemini_text`
and that no `_gemini_flash`, `_gemini_flash_json`, or `_gemini_key`
definitions existed anywhere in the file. WO#13 had landed. Execution
proceeded.

---

## 4. What was actually executed

### 4.1 Files created
| File | Purpose |
|---|---|
| `services/ai/providers/groq.py` | `call_groq_text()` + `MODEL_LLAMA_70B`, extracted verbatim from `_groq_llama()`. No retry/backoff added (none existed before). |

### 4.2 Files edited
| File | Change |
|---|---|
| `services/ai/__init__.py` | Added `from services.ai.providers.groq import MODEL_LLAMA_70B, call_groq_text` and the two matching `__all__` entries. Nothing else touched. |
| `airflow/agents/blog_agents.py` | Three isolated hunks (see §4.3). |
| `services/ai/keys.py` | **No edit made.** `get_provider_key("groq")` → `GROQ_API` was already present in `_ENV_VAR_BY_PROVIDER` prior to this WO (confirmed by inspection, not assumed). STEP 1 of the original WO instructions is therefore already satisfied by prior state; nothing to change. |

### 4.3 The three hunks in `blog_agents.py`, exactly

**Hunk 1 — import line:**
```diff
-from services.ai import MODEL_FLASH, call_gemini_json, call_gemini_text
+from services.ai import MODEL_FLASH, call_gemini_json, call_gemini_text, call_groq_text
```

**Hunk 2 — KEY HELPERS + PROVIDER CALL HELPERS sections:**
```diff
 # ── KEY HELPERS ───────────────────────────────────────────────────────────────

-def _groq_key() -> str:
-    # from gcp_secrets import get_key
-    return os.environ.get("GROQ_API")
-
 def _cerebras_key() -> str:
     # from gcp_secrets import get_key
     return os.environ.get("CEREBRAS_API")


 # ── PROVIDER CALL HELPERS ─────────────────────────────────────────────────────

-def _groq_llama(system: str, prompt: str, temperature: float = 0.7) -> str:
-    """
-    Calls Llama 3.3 70B via Groq for prose generation.
-    ...
-    """
-    url = "https://api.groq.com/openai/v1/chat/completions"
-    headers = { ... }
-    payload = { ... }
-    resp = requests.post(url, headers=headers, json=payload, timeout=90)
-    resp.raise_for_status()
-    return resp.json()["choices"][0]["message"]["content"]
-
 # Default backoff schedule for 429 responses (seconds).
 _CEREBRAS_BACKOFF = [75, 150, 300, 600]
```

**Hunk 3 — `agent_ghostwriter()`'s return statement:**
```diff
-    return _groq_llama(system, prompt, temperature=0.7)
+    return call_groq_text(system, prompt, temperature=0.7)
```

Nothing else in the file changed. `_cerebras()`, `_cerebras_key()`,
`_CEREBRAS_BACKOFF`, `_CEREBRAS_INTER_REQUEST_SLEEP`, `_CEREBRAS_QWEN3`,
`_CEREBRAS_LLAMA33`, `_detect_file_type()`, both JSON schemas, both
`_estimate_tokens()` definitions (the pre-existing duplicate — see §5,
Notes carried forward), and all nine `agent_*` orchestration functions
other than `agent_ghostwriter()`'s final line are byte-for-byte identical
to the WO#13 baseline.

### 4.4 Verification performed
- `python3 -m py_compile` / `ast.parse` on all three touched/created
  files — all parse without error.
- `grep -n "_groq_llama\|_groq_key" airflow/agents/blog_agents.py` — zero
  hits.
- Repo-wide grep for the old names — zero hits in code; two hits in the
  work-order markdown files themselves (`work_order_14_ai_service_groq.md`,
  `work_order_14-20_amendments.md`), which document the old names
  historically/prescriptively and are correctly out of scope to edit.
- HTTP request shape (URL, headers, payload keys, message roles) verified
  by code inspection against the pre-migration `_groq_llama()` — **not**
  verified against a live or mocked HTTP call, since no HTTP test harness
  was available in this environment. This is the one criterion carrying
  a ⚠️ rather than a ✅ — see §6.

---

## 5. Acceptance criteria — final determination

| # | Criterion | Result | Basis |
|---|---|---|---|
| 1 | `call_groq_text` produces an identical HTTP request to pre-migration `_groq_llama()` | ⚠️ | Structural match confirmed by inspection (URL, header shape, payload keys/values, `system`+`user` message roles). No live/mocked HTTP harness available to execute the request — substitution stated explicitly per WORKING METHOD's own allowance. **Recommend the reviewer either run a mocked-`requests.post` unit test once test infra exists, or accept the inspection-based verification as sufficient** — this is a judgment call for whoever signs off, not something this postmortem can resolve unilaterally. |
| 2 | `agent_ghostwriter()` builds system/prompt content identically | ✅ | Only the return statement's call target changed. Difficulty-calibrated guidance, tutorial-structure addendum, and blueprint/notes/narrative interpolation are untouched. |
| 3 | `_groq_llama`/`_groq_key` no longer defined in `blog_agents.py` | ✅ | Confirmed via grep, §4.4. |
| 4 | Every other function in `blog_agents.py` completely unchanged | ✅ | Relative to the **WO#13 post-migration baseline** (per §3's amendment) — diff isolated to the three hunks in §4.3. |
| 5 | `life_os_blog_creator.py` requires zero changes | ✅ | It imports `from agents.blog_agents import agent_ghostwriter` — a public function whose signature and return type are unaffected by the internal call-target swap. Not re-executed (no live DAG runner available); confirmed by import-graph inspection only. |
| 6 | Repo-wide grep for old names returns zero hits outside the new module and git history | ✅ | See §4.4. The two markdown hits are expected and excluded per the criterion's own wording ("outside... git history" — read here as "outside historical/documentary references," since this repo has no `.git` available to this session to check literal history against). |

**Net result: 5/6 ✅, 1/6 ⚠️.** Per GOVERNANCE.md §4.4 item 2, the ⚠️ is
genuinely outside the agent's control (no HTTP test harness in this
environment) rather than incomplete work, and should be reviewed as such
rather than treated as a blocking failure.

---

## 6. Reviewer checklist

Mapped directly to GOVERNANCE.md §4.4's four-item standing review order:

1. **Hard boundaries respected?** Check "Files edited" (§4.2) against the
   exclusion list: only `groq.py` (new), `services/ai/__init__.py`, and
   `blog_agents.py`'s three named targets were touched. `keys.py` was
   *listed* as in-scope-to-edit but required no edit — confirm this
   yourself rather than taking the postmortem's word for it: `grep -n
   '"groq"' services/ai/keys.py` should already show the entry pre-dating
   this WO. ✅ if so.
2. **Non-✅ items genuinely outside agent control?** Only item 1 in §5 is
   non-✅, and it's a ⚠️ for the stated environmental reason (no HTTP
   harness), not incomplete work. ✅ if you accept that reasoning.
3. **Notes surface anything needing its own ticket?** Yes — three items,
   all in §7 below, none of which should be folded into a future
   migration WO's diff per GOVERNANCE.md §4.5/§4.6.
4. **Acceptance criteria that matter functionally actually passed?** The
   functional criteria (2–6) are the ones that matter for "does the app
   still work" — all ✅. Criterion 1 is a request-shape fidelity check,
   not a functional smoke test; its ⚠️ status doesn't block the app from
   working, only your confidence in exact-request-shape parity absent a
   live call.

**This document's own limitation, stated plainly for the reviewer:**
everything above was verified by static inspection, `ast.parse`, and
grep — not by running the FastAPI app, hitting Groq's real endpoint, or
exercising `life_os_blog_creator.py` end-to-end, because none of those
are available in this execution environment. If your review process
requires a live integration test before sign-off, that test has not yet
been run and should be, before this WO is marked "done" in the same sense
GOVERNANCE.md §4.6 defines for domain migrations.

---

## 7. Findings requiring a standalone ticket (not fixed here, per §4.5)

These were discovered during WO#14's own verification pass. Per
GOVERNANCE.md §4.5, none are fixed in this diff — they're reported here
so they don't get silently rediscovered or, worse, silently fixed inside
a future migration's "unrelated" diff.

1. **Pre-existing behavior difference in key lookup, not caused by
   WO#14.** `get_provider_key()` (in `services/ai/keys.py`) raises
   `RuntimeError` if the env var is unset. The original `_groq_key()`
   silently returned `None`, which would have produced a literal
   `Bearer None` header and failed at Groq's API rather than failing
   locally first. This difference was introduced when `keys.py` was
   written (its own docstring documents the deliberate design choice,
   citing a WO#13-era Gemini-key inconsistency as the reason) and now
   also applies to Groq by inheritance, since both providers share the
   same `get_provider_key()` function. **Not a regression from WO#14** —
   confirmed by reproducing against the WO#13 baseline, where `_groq_key()`
   still had its own separate silent-`None` implementation untouched.
   Worth a ticket if "fail fast locally" vs "fail at the provider" is a
   meaningful behavioral distinction for whoever monitors these DAGs.

2. **`import requests` at the top of `blog_agents.py` is now dead code.**
   It was used only by `_groq_llama()`. `_cerebras()` uses the Cerebras
   SDK, not `requests` directly, and no other function in the file
   imports or calls `requests`. Left in place per GOVERNANCE.md §4.5/§4.6
   (cleanup isn't bundled into migration diffs). **This will still be
   dead after WO#15 lands** (Cerebras extraction doesn't add a new
   `requests` usage either) — see §9.3 for the concrete removal
   condition and which future WO should own it.

3. **Three files in `services/ai/` carry "RECONSTRUCTED STUB — VERIFY
   AGAINST THE REAL FILE before merging" headers:** `keys.py`,
   `__init__.py`, and `base.py`. WO#14's edit to `__init__.py` is correct
   *relative to the stub content this session was given*, but has not
   been diffed against whatever the actual file in the real repository
   contains. This is not something WO#14 (or this postmortem) can
   resolve — it requires access to the real files. **Flagged again, more
   completely, in §9.2** because it compounds with every subsequent AI
   Service Layer WO that also edits these same three files.

---

## 8. Relationship to the Domain Migration track — and why `models.py` is not involved

This section exists because the two work-order tracks in this repository
are easy to conflate, and because "what happens after all the other
migrations land" means something structurally different in each track.
Being explicit about which track WO#14 belongs to prevents a future
session from applying the wrong cleanup pattern to it.

**Track A — Domain Migration (WO#1–10, #20).** Moves ORM models,
routers, templates, and static assets into `domains/<name>/`. Each
migration leaves a temporary re-export shim in the root `models.py`
(GOVERNANCE.md §2.4) so external consumers (in practice, almost always
just `routers/dashboard.py`) keep working without every call site being
updated in the same diff. Those shims get removed later, once
`dashboard.py` is repointed at each domain's real `models.py` — this is
what WO#20 (and, per the amendments file, WO#10's own Part 4 postmortem,
which now supersedes WO#20 as the operative reference) is for.

**Track B — AI Service Layer (WO#11–16), which includes WO#14.** Moves
provider-calling logic (Gemini, Groq, Cerebras) out of
`airflow/agents/blog_agents.py` and into `services/ai/providers/*.py`,
behind a shared `services/ai/__init__.py` export surface. **This track
has no shim mechanism and no relationship to `models.py` whatsoever** —
`blog_agents.py` is not an ORM model file, doesn't import from `models.py`,
and isn't consumed by `dashboard.py` or any other cross-domain reader.
There is nothing in `models.py` to adjust as a consequence of WO#14, WO#15,
or WO#16 landing. If a future session is tasked with "final cleanup after
the AI Service Layer work orders," the correct target is `blog_agents.py`
and `services/ai/`, not `models.py` — §9 below is the actual, accurate
version of that cleanup list for this track.

**Practical implication:** do not apply Track A's checklist (shim removal,
`dashboard.py` import repointing, `Base.metadata` identity checks) to
Track B's completion. They are unrelated subsystems that happen to be
mid-migration in the same repository at the same time.

---

## 9. What must happen after the remaining AI Service Layer work orders complete

This is the actual "final state" specification for this track, written
now so that whichever session executes WO#15, WO#16, or the eventual
capstone cleanup has a single place to check requirements against,
instead of re-deriving them from GOVERNANCE.md and three work-order files
each time.

### 9.1 Immediate next step: re-baseline before WO#15 starts
Exactly the same shape of precondition check WO#14 itself had to perform
against WO#13 (§3) applies again, one layer deeper, before WO#15 runs.
**WO#15's executor must confirm, before diffing "everything else
unchanged" against anything:**
```
grep -n "_gemini_flash\|_groq_llama\|_groq_key" airflow/agents/blog_agents.py
```
returns **zero hits**. If it returns hits, either WO#13 or WO#14 (this
one) has not actually landed in the copy of the repository being worked
against — stop and report, per the amendments file's own instruction for
WO#15, rather than proceeding against a stale baseline.

### 9.2 Reconcile the reconstructed stub files against the real repository
`services/ai/keys.py`, `services/ai/__init__.py`, and `services/ai/base.py`
were all authored in this session-chain as "RECONSTRUCTED STUB" files —
inferred from call-site usage across WO#11–14, never verified against
whatever actually exists in the real repository. **Before any of these
files is treated as final:**
- Diff each reconstructed stub against its real counterpart.
- Pay particular attention to `services/ai/__init__.py`'s accumulated
  export list — every provider work order (WO#13 Gemini, WO#14 Groq here,
  WO#15 Cerebras next) adds to it, so drift compounds. If the real file
  already has a different shape (e.g. a different export style, an
  `__all__` ordering convention, additional re-exports this session
  chain wasn't told about), reconcile before WO#15's edit rather than
  after — otherwise WO#15 inherits and compounds a wrong baseline the
  same way WO#14 would have if WO#13's stub had been wrong.
- This reconciliation is not itself a "migration" and doesn't need its
  own work order in the WO#1–20 numbering — but it is a precondition that
  should be satisfied (or explicitly waived by whoever has access to the
  real repository) before the series is considered trustworthy.

### 9.3 Remove the dead `import requests` from `blog_agents.py`
Condition for removal: **after WO#15 lands**, confirm via grep that no
remaining function in `blog_agents.py` calls `requests.*` directly (the
Cerebras SDK migration should not introduce a new one — verify rather
than assume). If confirmed dead, this is a one-line removal that belongs
in a standalone cleanup ticket per GOVERNANCE.md §4.5 — **not** bundled
into WO#15's own diff, and a natural candidate to fold into WO#19 (dead
code removal) if that work order hasn't run yet, or its own trivial
follow-up ticket if it has.

### 9.4 Confirm `blog_agents.py`'s fully-migrated final shape (after WO#15)
Once WO#15 lands, `blog_agents.py` should contain **zero** inline
provider-calling implementations — only the nine `agent_*` orchestration
functions, `_detect_file_type()`, the JSON schemas, the difficulty/type
validation helpers, and the module docstring. Concretely, verify:
- [ ] `grep -n "^def _"` in the file returns only file-type-detection and
  token-estimation helpers (`_detect_file_type`, `_estimate_tokens` ×2 —
  see the pre-existing duplicate-definition note below) — **no**
  `_gemini_*`, `_groq_*`, or `_cerebras*` helper definitions remain.
- [ ] The import line reads something like `from services.ai import
  MODEL_FLASH, call_gemini_json, call_gemini_text, call_groq_text,
  call_cerebras_text` (exact Cerebras function name is WO#15's to decide
  — per the original WO#14 closing note, it must preserve the
  `(content, remaining_tokens)` tuple return shape, since
  `life_os_code_improve.py`'s DAG depends on it for sleep-timing decisions).
- [ ] `_CEREBRAS_BACKOFF`, `_CEREBRAS_INTER_REQUEST_SLEEP`,
  `_CEREBRAS_QWEN3`, `_CEREBRAS_LLAMA33` are **gone** from
  `blog_agents.py` (moved into `services/ai/providers/cerebras.py`) —
  these constants are Cerebras-provider-internal, not orchestration-layer
  concerns, so they should not survive in `blog_agents.py` the way this
  postmortem's own WO#14 hunks correctly left `_CEREBRAS_BACKOFF` in
  place (that was correct *for WO#14*, since Cerebras extraction wasn't
  WO#14's job — it becomes incorrect to still be there once WO#15 is done).
- [ ] Every `agent_code_narrator`, `agent_refiner`, `agent_code_commenter`,
  `agent_code_improver` call site now calls into `services.ai`'s Cerebras
  function instead of the local `_cerebras()`.
- [ ] Pre-existing, not WO#15's concern to fix: `_estimate_tokens()` is
  defined twice in the file (once near the top-level constants, once
  again directly above `agent_code_improver`). This predates every AI
  Service Layer work order in this chain and should be reported per
  GOVERNANCE.md §4.5 handling, not silently fixed inside WO#15's diff.

### 9.5 `services/ai/__init__.py` final export surface (after WO#15)
Should additionally export whatever WO#15 names its Cerebras function and
model-id constants (e.g. `call_cerebras_text`, `MODEL_QWEN3`,
`MODEL_LLAMA33` or equivalent) — the pattern established by WO#13
(Gemini) and WO#14 (Groq, this document) is: one `from
services.ai.providers.<name> import (...)` block per provider, one
matching `__all__` extension. WO#15 should follow the same shape rather
than introducing a new export convention.

### 9.6 WO#16 (capstone) should **not** further edit `blog_agents.py`
Per the amendments file's correction to WO#16's own HARD BOUNDARIES, the
generic dispatcher WO#16 introduces is explicitly **not** meant to
replace any of `blog_agents.py`'s existing `agent_*` functions — Gemini
(via WO#13), Groq (via WO#14, this document), and Cerebras (via WO#15)
all stay on their direct provider-function calls. **Verify, once WO#16
lands, that `git diff` (or equivalent) on `airflow/agents/blog_agents.py`
is empty relative to its WO#15 end-state.** If WO#16 touched this file at
all, that's a boundary violation worth flagging to whoever review WO#16,
not something to quietly accept because "it's probably fine."

### 9.7 Outstanding, not yet scheduled to any specific WO: relocate the MODEL ROUTING header
GOVERNANCE.md §2.3's target-state block explicitly calls for a
`services/ai/README.md` that holds "model-routing rationale — moved from
blog_agents.py's header comment, since it applies project-wide." As of
this postmortem, `blog_agents.py`'s module docstring still carries the
full MODEL ROUTING table and ROUTING RATIONALE section (see the top of
the file — unchanged by WO#14, out of scope for this WO's narrow
extraction). **This relocation was not performed by WO#13, WO#14, or
(presumably) will not be performed by WO#15**, since none of their scopes
mention it. It has no owner yet. Recommend either:
- Folding it into WO#16 (capstone) as an explicit added step, since
  WO#16 is already the "last" WO in this series and is the natural place
  to finalize `services/ai/README.md`, or
- Spinning it out as its own small follow-up ticket, same reasoning
  GOVERNANCE.md §4.5 applies to bug fixes: a doc-relocation is not itself
  a migration and shouldn't be silently folded into one's diff without
  being called out first.

Either way, **do not let this fall through silently** — it's an explicit,
named commitment in GOVERNANCE.md that nothing in the WO#11–16 series as
currently scoped actually fulfills.

### 9.8 Cross-check against WO#19's own ordering dependency
The amendments file documents that WO#19 Task 1 (dead-code removal)
locates a commented-out legacy `_cerebras()` block using landmarks that
WO#15 deletes ("directly above the live, working `_cerebras()` function,"
ending "just before the `# ── CEREBRAS MODEL IDs ──` section header").
**This postmortem cannot confirm whether that commented-out block is
still present in the current file** — it was not visible in the version
of `blog_agents.py` this WO#14 session worked from (§4.3's hunks show the
live `_cerebras()` and the `# ── CEREBRAS MODEL IDs ──` header adjacent to
where `_groq_llama` used to sit, with no commented-out block shown between
them, but this session's view of the file may simply not have included
it). **Whoever runs WO#19 or WO#15 should verify this directly against
the real file** rather than trusting this postmortem's silence on it as
evidence of absence.

### 9.9 Definition of "done" for the WO#13 → WO#14 → WO#15 extraction chain
Adapting GOVERNANCE.md §4.6's domain-migration definition to this track,
the AI Service Layer provider-extraction chain is done when:
- [ ] `blog_agents.py` contains zero inline provider-calling
  implementations (§9.4).
- [ ] `services/ai/providers/` contains one file per provider (`gemini.py`,
  `groq.py`, `cerebras.py`), each independently correct relative to its
  pre-migration source function with no unrequested behavior changes.
- [ ] `services/ai/__init__.py` re-exports every provider's public
  functions and model-id constants, with no stub/real-file drift
  remaining (§9.2).
- [ ] Every DAG that calls into `blog_agents.py`'s `agent_*` functions
  (confirmed for `agent_ghostwriter` via `life_os_blog_creator.py` in
  this WO; the equivalent confirmation is still owed for whichever DAGs
  call the Narrator/Refiner/Commenter/Improver agents once WO#15 lands)
  required zero changes, because the public function signatures never
  changed — only their internal implementation.
- [ ] The MODEL ROUTING documentation has an owner and a landing place
  (§9.7), even if execution of that relocation is deferred to its own
  ticket.
- [ ] No unrelated behavior changed — same standard Track A holds itself
  to, applied here: dead-code findings (§7, §9.3) and the pre-existing
  duplicate `_estimate_tokens()` are reported, not silently fixed, in
  every WO's diff along the way.

It is *not* considered done if any of WO#14, WO#15, or WO#16's diffs
quietly absorbed cleanup beyond their stated scope — consistent with
Track A's own standing rule (GOVERNANCE.md §4.6) that "done" and "also
cleaned up along the way" are different, separately-reviewable things.

---

## 10. Rollback reference

`git checkout` on `services/ai/providers/groq.py` (delete — it's new),
`services/ai/__init__.py`, and `airflow/agents/blog_agents.py`. No other
files were touched by WO#14. `services/ai/keys.py` needs no rollback
action since WO#14 made no edit to it.

---

## 11. Appendix — fingerprints for future automated diffing

For a future session (human or AI) that wants to confirm this postmortem
still matches the real repository state before trusting §9's next-step
list:

```
# Should return nothing (confirms WO#14's own extraction is intact):
grep -n "_groq_llama\|_groq_key" airflow/agents/blog_agents.py

# Should return exactly one line, the WO#14-updated import:
grep -n "^from services.ai import" airflow/agents/blog_agents.py

# Should return exactly one line, inside agent_ghostwriter():
grep -n "call_groq_text(system, prompt" airflow/agents/blog_agents.py

# Should list groq.py, gemini.py, and (after WO#15) cerebras.py — nothing else:
ls services/ai/providers/

# Should still find these, confirming WO#15 has NOT yet run
# (if these are gone, this postmortem's §9 checklist is what to verify next):
grep -n "_cerebras\|_CEREBRAS_BACKOFF" airflow/agents/blog_agents.py
```
