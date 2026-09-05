# WO#16 — AI Service Layer Capstone — Postmortem OVERVIEW & Full-Program Closeout Requirements

**Purpose of this document, specifically:** the detailed technical
postmortem (`WO16-domains-migration-postmortem-ai-capstone.md`) already
covers diffs, mocked-request verification, and line-level acceptance
criteria. **This document is different in kind, not a duplicate.** It
exists for two audiences:

1. **The reviewer deciding whether WO#16 passes** — Part A/B/C below
   separate, explicitly, "what WO#16's own text asked for as originally
   drafted" from "what was agreed to change before a line of code was
   touched" from "what was actually done" — so a pass/fail judgment can
   be made against the right baseline, not a moving target.
2. **Whichever agent picks up cleanup once every other migration in this
   program (not just AI Service Layer) is also done** — Part G is a
   single, consolidated closeout checklist that pulls together the
   scattered "post-migration follow-up" sections buried in the WO#11
   through WO#15 postmortems, resolves the `models.py` question
   explicitly and thoroughly, and flags a stale-documentation problem
   this session found while assembling it.

**Read Part G before doing any "final cleanup" work on this repository.**
It is written to be load-bearing on its own, without requiring a
re-read of all five prior postmortems first.

---

## Part A — What WO#16's Own Text Specified, As Originally Drafted

Preserved here unamended, exactly as WO#11 through WO#15's postmortems
each did for their own work order, so a reviewer can see precisely what
changed between "drafted" and "executed" without diffing two source
documents by hand.

- **ROLE:** senior refactoring engineer closing out a multi-phase
  consolidation effort — add the generalized interface now that three
  real provider shapes exist, close the vision capability gap, make one
  deliberate documented scope decision on `finance_upload.py`. Explicitly
  **not** a license to touch any of the five callers already migrated in
  WO#11–15.
- **HARD BOUNDARIES (as originally drafted):**
  - Do not migrate `job_agents.py`, `recipe_agents.py`, `weekly_agents.py`,
    **`workout_plans.py`**, `media_recommend.py`, or
    `blog_agents.py`'s Ghostwriter/Narrator/Refiner/Commenter/Improver
    functions onto the new generic dispatcher.
  - The generic `call_ai_text()` normalizes away Cerebras's
    `remaining_tokens` — document, don't design around.
  - `finance_upload.py`'s SDK call is explicitly not migrated.
  - Vision support scoped narrowly to matching
    `agent_extract_recipe_from_image()`'s exact shape.
- **SCOPE (as originally drafted):** edit `services/ai/__init__.py`,
  `services/ai/providers/gemini.py`; create `services/ai/README.md`;
  edit `recipe_agents.py` (one function only); edit
  `routers/finance_upload.py` **(or `domains/finance/routers/finance_upload.py`
  if WO#5 has already run)**, comment only.
- **STEPS (as originally drafted):** (1) design `call_ai_text`/`call_ai_json`
  in `services/ai/__init__.py`; (2) confirm backward compatibility;
  (3) add `call_gemini_vision_json()` to `gemini.py`, migrate the one
  recipe function; (4) document the `finance_upload.py` decision with a
  comment; (5) create `services/ai/README.md` with the routing table
  (copied, not moved, from `blog_agents.py`), usage guidance, the SDK
  decision, and a provider-coverage list.
- **ACCEPTANCE CRITERIA (as originally drafted):** dispatcher routes
  correctly per provider and raises `ValueError` on an unknown one;
  vision request is identical to the pre-migration raw implementation;
  post-processing on the vision function unchanged; **"confirm zero
  changes" to `job_agents.py`, `weekly_agents.py`, `workout_plans.py`,
  `media_recommend.py`, and every `blog_agents.py` function**;
  `recipe_agents.py`'s other four functions unchanged; `finance_upload.py`
  comment-only; README covers all four sections; `blog_agents.py`'s
  docstring routing table still present (not deleted).

At the time WO#16 was originally drafted, its own "confirm zero changes"
criterion and its SCOPE's conditional file paths (`routers/finance_upload.py`
**or** the `domains/` equivalent) both assumed a state of the repository
that the amendment below corrects.

---

## Part B — What Was Agreed to Change Before Execution (the Amendment)

Three corrections, none of which change Task 1's actual dispatcher/vision/
finance_upload.py work — they change the *baseline* that work is measured
against and add one **new, separately-scoped task**.

### B.1 — File rename correction

`workout_plans.py` → `workout_plan_ai_generator.py` everywhere it appears
in WO#16's text (HARD BOUNDARIES, SCOPE's "confirm zero changes" list,
ACCEPTANCE CRITERIA). Reason: WO#8's own authorized follow-up work split
the original `workout_plans.py` into `workout_plans_crud.py` and
`workout_plan_ai_generator.py` before WO#16 was ever drafted — the
original file this WO#16 draft referenced no longer exists under that name.

### B.2 — Mandatory pre-execution gate (new, before Step 1)

Required, before touching any file: confirm `services/ai/base.py`,
`services/ai/keys.py`, and `services/ai/__init__.py` against the real
repository, since all three have been reconstructed stubs — never
verified — across WO#11, #13, #14, #15. If the real files still aren't
available, **state that explicitly as an unresolved ⚠️ carried forward,
not silently inherited.** Separately: verify every "unchanged" claim
independently by diffing against actual prior state, rather than
trusting a file's own "included unmodified for context" label — WO#15's
own deliverable shipped five silently-truncated files under exactly that
label, caught only after a direct follow-up question.

### B.3 — New Task 2: `blog_agents.py` shell-level cleanup (new scope)

Confirmed as real, owned-by-nobody findings in both the WO#14 and WO#15
postmortems, folded into WO#16 as an explicit, separately-labeled second
task (mirroring WO#19's own precedent for keeping two kinds of work
diff-separable within one work order): remove the now-dead `import os`
and `import time` (confirm `import requests`, already flagged dead by
WO#14, is still dead); remove the now-empty `# ── KEY HELPERS ──` header;
resolve (confirm-before-deleting, not assume-and-delete)
`_CEREBRAS_INTER_REQUEST_SLEEP`; fold the MODEL ROUTING table relocation
into the same README.md Step 5 already creates; **do not touch** the
duplicate `_estimate_tokens()` definition — report, per standing policy.

**No change** to the vision-support work or the `finance_upload.py`
documentation decision — both proceed exactly as Part A specified.

---

## Part C — What Was Actually Executed

*(Full diffs, mocked-request-shape verification, and dispatcher-routing
test output live in `WO16-domains-migration-postmortem-ai-capstone.md`
— summarized here for the reviewer's convenience, not restated in full.)*

**Task 1 (Part A's scope, corrected per B.1):**
- `services/ai/__init__.py` — added `call_ai_text()`, `call_ai_json()`,
  re-exported `call_gemini_vision_json`. All 11 pre-existing exports
  confirmed identical by object-identity check (not visual comparison).
- `services/ai/providers/gemini.py` — added `call_gemini_vision_json()`.
  One flagged, explained deviation from the literal text: added an
  optional `timeout: float = 120.0` parameter not present in WO#16's own
  Step 3 signature, because omitting it would have silently dropped the
  vision call's timeout from 120s to `post_with_retry`'s 90s default —
  contradicting WO#16's own request-equivalence acceptance criterion.
- `airflow/agents/recipe_agents.py` — only
  `agent_extract_recipe_from_image()` and its import block changed;
  request-shape equivalence verified against a mocked HTTP layer using
  the function's real, unmodified system prompt and schema (not
  placeholder text).
- `domains/finance/routers/finance_upload.py` — comment-only, verified
  by stripping comment lines from both versions and confirming the
  remainder is byte-identical.
- `services/ai/README.md` — created, all four required sections present.

**Task 2 (B.3's new scope):**
- `airflow/agents/blog_agents.py` — top-of-file region only: three dead
  imports removed, one empty section header removed,
  `_CEREBRAS_INTER_REQUEST_SLEEP` annotated but **not deleted** (see
  Part B.3). Verified via a real, full 1,452-line byte-exact
  reconstruction of the file (not a partial stand-in) diffed against the
  edited version — exactly two hunks, nothing else touched. A repo-wide
  (well, file-wide) `grep`-equivalent for `os.`/`time.`/`requests.`
  usage across the complete file returned zero hits outside the removed
  import lines themselves, upgrading an earlier "manual review only"
  claim in the first postmortem draft to a genuinely sandbox-verified one.

**A correction surfaced during Task 2, not previously reported:** the
WO#14/#15 postmortems' own "definition of done" checklists state that
`grep -n "^def _"` in `blog_agents.py` should return only
`_detect_file_type` and the duplicate `_estimate_tokens`. Run against
the real, complete file, it also returns `_summarize_recent_difficulties`
and `_validate_difficulty_distribution` — two legitimate, always-present,
non-provider Researcher-agent helpers that were simply never accounted
for in that inherited checklist. Not a code problem; a documentation
correction owed to those two postmortems.

---

## Part D — Verification Method (pointer, not restatement)

Every claim in Part C above was checked by actually running code in a
sandboxed environment — a fake `requests.post` capturing request shape,
monkeypatched provider functions to test dispatcher routing, `ast.parse`
on every touched file, and `diff`/comment-stripped-diff against
byte-exact reconstructions of the real source — not by reading the code
and asserting it looked right. Full transcripts, capture output, and the
specific test assertions are in the companion technical postmortem.
No live Gemini/Groq/Cerebras API call was made; this is stated as an
open ⚠️ there, not hidden here.

---

## Part E — Acceptance Criteria, Final Determination

| # | Criterion | Result | Note |
|---|---|---|---|
| 1 | `call_ai_text` dispatches correctly per provider; Cerebras discards `remaining_tokens` | ✅ | Real monkeypatched execution |
| 2 | `call_ai_text(provider="nonexistent")` raises `ValueError` | ✅ | |
| 3 | `call_ai_json(provider="gemini", ...)` dispatches to `call_gemini_json` | ✅ | Gemma-model routing to `call_gemma_json` also verified |
| 4 | Vision request identical to pre-migration raw implementation | ✅ | Full payload incl. real system prompt/schema, not placeholders |
| 5 | Vision function's post-processing/exception-fallback unchanged | ✅ | |
| 6 | Zero changes: `job_agents.py`, `weekly_agents.py`, `workout_plan_ai_generator.py` *(corrected per B.1)*, `media_recommend.py` | ✅ | Not touched — omission confirmed, not just SCOPE-asserted |
| 6b | Zero changes to `blog_agents.py`'s agent functions/logic | ✅ | Task 2's diff is isolated to imports + one header |
| 6c | *(superseded)* Original "zero changes to `blog_agents.py`" wording | ⚠️ superseded, not failed | The amendment's Task 2 explicitly authorizes the narrow exception in 6b; criterion 6 (the other four files) stands as originally worded |
| 7 | `recipe_agents.py`'s other four functions unchanged | ✅ | |
| 8 | `finance_upload.py` byte-for-byte behavior unchanged | ✅ | Comment-stripped diff, not just visual diff |
| 9 | `services/ai/README.md` covers all four required sections | ✅ | |
| 10 | `blog_agents.py`'s docstring routing table still present | ✅ | Not in Task 2's touched region |

**10/10 applicable, 1 explicitly superseded (not failed) by an
authorized amendment — see row 6c.**

---

## Part F — Reviewer Sign-Off Checklist

Mapped to GOVERNANCE.md §4.4's standing four-item review order, applied
here specifically:

1. **Hard boundaries respected?** Check Part C's "Files edited" against
   Part A/B's exclusion list. `job_agents.py`, `weekly_agents.py`,
   `workout_plan_ai_generator.py`, `media_recommend.py` — absent from
   every diff, confirmed. `blog_agents.py` — touched, but only under
   Task 2's explicit, amendment-authorized exception (row 6c); confirm
   the diff shown in Part C's Task 2 section is the *only* change to
   that file, nothing under Task 1.
2. **Non-✅ items genuinely outside this session's control, not
   incomplete work?** Only row 6c is non-✅, and it's a documented
   supersession, not a gap — the amendment itself authorized it before
   execution began. No item in Part E is a ⚠️/❌ caused by something this
   session failed to do.
3. **Does anything in Notes need its own ticket, filed separately?**
   Yes — four items, all listed in the companion postmortem's §5 and
   restated in Part G.5 below. None should be folded into a future
   migration's diff without being called out first, per GOVERNANCE §4.5.
4. **Do the acceptance criteria that matter functionally actually
   pass?** Yes — every criterion with real functional weight (dispatcher
   correctness, request-shape equivalence, zero-change confirmations)
   passed under real execution, not inspection. The one non-✅ is a
   documentation/scope bookkeeping item, not a functional gap.

**This session's own stated limitation, repeated here deliberately:**
no live API call, no live FastAPI/SQLAlchemy stack, and the real
`services/ai/base.py`/`keys.py`/`__init__.py` were never available to
check against. If your sign-off standard requires closing those gaps
first, they are not closed by this work order and should not be treated
as implicitly resolved — see Part G.1 below for exactly what to do about
this.

---

## Part G — Post-Program Closeout: What Must Happen After All Other Migrations Are Completed

**This section is written to stand alone.** A future agent should be
able to use it as a requirements checklist without first re-reading
WO#11 through WO#16's individual postmortems — though citations back to
them are included for anyone who wants the original reasoning.

### G.0 — Two independent tracks, restated plainly (read this before touching anything)

This repository has **two unrelated migration efforts running
concurrently**, established as early as the WO#14 postmortem (§8) and
reconfirmed at every subsequent work order:

- **Track A — Domain Migration** (WO#1–10, plus the shim-removal pass
  WO#20): moves ORM models, routers, templates, and static assets into
  `domains/<name>/`. Owns `models.py` — both the root shim file and each
  domain's own `models.py`.
- **Track B — AI Service Layer** (WO#11–16, **now complete** as of this
  document): moves LLM provider-calling logic into `services/ai/`. Has
  **no shim mechanism and no relationship to `models.py` whatsoever** —
  none of its six work orders import from, edit, or have any reason to
  reference any `models.py` file, root or domain-specific.
- **Track C — Frontend Consolidation** (WO#17): toast notification
  dedup, dead `sidebar_js.html` removal. Independent of A and B.
- **Track D — DAG Reorganization** (WO#18): relocates
  `airflow/dags/*.py` into domain subfolders. Independent of A, B, C.
- **Track E — Misc Cleanup** (WO#19): `blog_agents.py` dead-code removal
  (the historical commented-out `_cerebras()` block, if it exists — see
  G.5) plus a router line-count lint script.

**Do not apply Track A's checklist (shim removal, `dashboard.py`
repointing, `Base.metadata` identity checks) to Track B's completion,
or vice versa.** They are unrelated subsystems that happen to be
mid-migration in the same repository at the same time. A request to
"finish cleanup on the AI stuff" means Track B (§G.3 below); a request
to "finish cleanup on the models" means Track A (§G.2 below) — treat
these as different requests even when phrased similarly.

### G.1 — Highest priority, blocks trusting anything else: replace the reconstructed `services/ai/` stub files

Repeated here as the single most load-bearing item in this entire
closeout, because it has now been repeated — and still not resolved —
across **six consecutive work orders** (WO#11, #13, #14, #15, #16, and
now this document):

`services/ai/base.py`, `services/ai/keys.py`, and `services/ai/__init__.py`
have never been supplied as real source material in any session across
this whole program. Every claim in every postmortem about their exact
behavior (`post_with_retry`'s retry/backoff timing, `get_provider_key`'s
exception-vs-`None` behavior on a missing key, `__init__.py`'s exact
export shape) is an inference from call-site usage, not a confirmed fact.

**Before this program is considered done, whoever has the real
repository must:**
1. Diff each of the three reconstructed files against its real
   counterpart.
2. Re-run every payload-shape and retry-path verification across the
   entire series against the real files: `job_agents.py` (WO#11),
   `recipe_agents.py`'s five call sites including vision (WO#12, WO#16),
   `weekly_agents.py`, `workout_plan_ai_generator.py`,
   `media_recommend.py`, `blog_agents.py`'s twelve call sites across
   Gemini/Groq/Cerebras (WO#13, #14, #15), and WO#16's dispatcher/vision
   additions.
3. Specifically re-run WO#15's 8 Cerebras retry-path scenarios against
   the **real, installed** `cerebras-cloud-sdk` package (currently
   pinned **unversioned** in `Dockerfile.airflow` — a version bump could
   silently change `.with_raw_response`'s shape and nobody in this
   program's history would know, since no session has ever run against
   the real SDK).
4. Confirm `get_provider_key()`'s fail-loud-on-missing-key behavior
   (assumed since WO#11, propagated into every provider's key lookup) is
   what the real file actually does.

Nothing else in this document should be treated as more urgent than this.

### G.2 — Track A: the `models.py` question, addressed thoroughly and directly

**This is the section explicitly requested — read all of it, not just
the conclusion, before telling anyone "there's nothing to do here."**

**G.2.1 — What has actually been read, and by whom, across this entire
program:**

| File | Was it ever provided as real source in this program? | Current known state |
|---|---|---|
| Root `models.py` | **Yes** — provided directly in this engagement (the document showing `# ── WO#20 shim-removal pass ──`) | Confirmed, by direct read, to be a post-WO#20 documentation stub: every domain's re-export shim already removed, ~10 now-unused top-level imports (`datetime`, `enum`, `math`, `Decimal`, `Optional`, SQLAlchemy column/type imports, `Mapped`/`mapped_column`/`relationship`, `BaseModel`) left in place on purpose, per WO#20's own narrow scope ("removing a shim and updating exactly one import line in `dashboard.py` per domain," not general dead-code cleanup — that broader cleanup is its own tracked, separate item, see G.2.4). **Zero references to Gemini, Groq, Cerebras, any API key, or `services.ai` anywhere in this file.** |
| `domains/media/models.py` | **No.** Never supplied in any work order across this entire program (WO#1–20). | Unknown. Every claim about `MediaItem` (its `media_type` field's exact representation, whether it's a plain string or a Python `Enum`, whether `"tv_show"` is the literal stored value) is an **inference** from how *other* files (`tmdb_service.py`, `media_recommend.py`, `media_agents.py`) use it — never confirmed against the model definition itself. |
| `domains/workout/models.py` | **No.** Same status. | Unknown. `WorkoutPlan`, `WorkoutPlanDay`, `Exercise`, etc. are used extensively by `workout_plan_ai_generator.py` and `weekly_agents.py`, but the model file itself has never been read. |
| `domains/finance/models.py` | **No.** Referenced by import in `finance_upload.py` (`Account`, `Category`, `Transaction`) but the file itself was never supplied. | Unknown. |
| `domains/recipes/models.py` | **No.** | Unknown — referenced only by inference from `recipe_agents.py`/`recipe_extract.py`/`recipe_discovery.py` usage. |
| Every other domain's `models.py` | **No**, none have ever been supplied. | Unknown. |

**G.2.2 — Direct answer to "does AI Service Layer need to touch
`models.py`":** No, and this is now a *confirmed* finding (per the root
file being directly read), not an assumption inherited from an earlier
postmortem's inference. The root `models.py` contains zero AI-provider
references. Nothing in WO#11–16's own scope ever touched it, imported
from it, or had a reason to. **If a domain's `models.py` file (media,
workout, finance, recipes, or any other) is eventually provided and
*does* turn out to contain an AI-provider reference** — a hardcoded
model ID, an inline `requests.post` call to a Gemini/Groq/Cerebras
endpoint, an import of `services.ai` — **that would not be a routine
cleanup edit.** It would mean either (a) someone added inline AI-calling
logic directly into an ORM model file, which is itself a standalone
GOVERNANCE.md §2.1 violation ("all AI integration logic must live in
`services/ai/`, never inline in a router, template, **or model**") worth
flagging as its own finding before fixing, or (b) this whole
two-independent-tracks premise has stopped holding for some
project-specific reason — significant enough to stop and report, per
this program's standing pre-existing-bug-handling discipline, rather
than silently edit it away.

**G.2.3 — What to actually do if instructed to "adjust `models.py`,
removing AI-service references from there":**
1. **Re-fetch the real, current `models.py`** (root, and whichever
   domain the instruction is actually about, if a domain is named) —
   do not answer from this document's snapshot, which could be stale by
   the time that instruction arrives.
2. **Confirm the zero-references finding directly** against that fresh
   copy, the same way this session did for the root file — a grep for
   `gemini`, `groq`, `cerebras`, `GEMINI_API`, `GROQ_API`, `CEREBRAS_API`,
   `services.ai`, and `services\.ai` (both dotted and underscored forms,
   in case of `from services import ai` style imports) is sufficient.
3. **If the grep comes back clean** (the expected, and so-far-always-true,
   outcome): report that directly — "confirmed no AI-service references
   in `<file>` as of `<date/commit>`; nothing to remove" — rather than
   either silently doing nothing without saying so, or inventing a
   change to satisfy the instruction's apparent expectation that
   something needs removing.
4. **If the grep finds something:** stop before editing. Determine
   whether it's (a) a genuine GOVERNANCE §2.1 violation needing its own
   flagged, separately-reviewed fix, or (b) evidence the file being
   checked isn't actually an ORM model file in the sense this document
   assumes (e.g., a differently-named file that also happens to be
   called "models.py" somewhere unexpected). Either way, report before
   acting — this is exactly the kind of finding this whole program's
   "report, don't silently fix" discipline exists for.

**G.2.4 — The *actual* remaining `models.py`-adjacent work, which
belongs to Track A, not Track B:** per GOVERNANCE.md §5 and the root
`models.py` file's own header comment, there is a real, tracked,
**Track A** follow-up: reduce the root `models.py` from its current
~10-line dead-import stub to either a clean, minimal import-registry
file, or delete it entirely and move the "importing every domain's
`models.py` registers it with the shared mapper" guarantee into
`database.py`'s `init_db()` instead (the two options WO#10's postmortem
§4.2 already names). **This is unrelated to AI Service Layer and was
never in scope for WO#11–16** — it's listed here only so a future agent
searching this document for "models.py" finds the real, actionable item
instead of assuming Track B owns it.

### G.3 — Track B (AI Service Layer): consolidated Definition of Done

Synthesized from WO#11 §3.9, WO#14 §9.9, WO#15 §6.11, and this work
order — one master checklist, superseding the three partial versions
scattered across those postmortems:

- [x] `job_agents.py` migrated (WO#11).
- [x] `recipe_agents.py` migrated, including vision (WO#12, completed WO#16).
- [x] `weekly_agents.py`, `workout_plan_ai_generator.py`,
      `media_recommend.py` migrated (WO#13).
- [x] `blog_agents.py`'s Gemini functions migrated (WO#13).
- [x] `blog_agents.py`'s Groq function migrated (WO#14).
- [x] `blog_agents.py`'s Cerebras functions migrated (WO#15).
- [x] `services/ai/providers/` contains `gemini.py`, `groq.py`,
      `cerebras.py`, each independently correct relative to its
      pre-migration source (per each WO's own verification).
- [x] Generic `call_ai_text()`/`call_ai_json()` dispatcher exists,
      informed by three real provider shapes (WO#16).
- [x] Vision support exists and the last raw-HTTP AI call in the
      non-SDK part of the codebase is migrated (WO#16).
- [x] `finance_upload.py`'s SDK exception is explicitly documented,
      not silently left unexplained (WO#16).
- [x] `services/ai/README.md` exists with routing rationale, usage
      guidance, the SDK exception, and provider coverage (WO#16).
- [ ] **`services/ai/__init__.py`, `base.py`, `keys.py` verified against
      real source.** Still open — see G.1. This is the one item blocking
      "done" from being unconditionally true.
- [ ] `blog_agents.py`'s few remaining shell-cleanup items — see G.5.
- [ ] Repo-wide sweep confirming no AI-provider reference exists outside
      `services/ai/` and the known caller files — never actually run as
      a full sweep in any single session; each work order confirmed only
      its own files.
- [ ] Test suite / CI audit — never performed in any session in this
      program (no test files were ever supplied). Before merging
      anything: search for references to every deleted symbol
      (`_gemini_flash`, `_gemini_flash_json`, `_gemini_key`, `_groq_llama`,
      `_groq_key`, `_cerebras`, `_cerebras_key`, `_CEREBRAS_BACKOFF`,
      `_CEREBRAS_QWEN3`, `_CEREBRAS_LLAMA33`) and any test asserting the
      old `call_gemini_json(system, prompt, schema, ...)` positional
      signature.
- [ ] GOVERNANCE.md §2.3 actually read and updated — every postmortem in
      this chain has cited it without anyone confirming its real text;
      see G.4 for the specific discrepancy that needs resolving there.
- [ ] Async-route-calls-sync-`services/ai/` audit — only two call sites
      (`workout_plan_ai_generator.py`, `media_recommend.py`) were ever
      fixed with `asyncio.to_thread(...)`, because they were the only two
      visible in any single session's file set. Never a codebase-wide
      audit. Any router anywhere in `domains/` calling
      `call_gemini_json`/`call_gemini_text`/`call_gemma_json`/
      `call_gemini_vision_json`/`call_ai_text`/`call_ai_json` from an
      `async def` handler needs the same check.

### G.4 — GOVERNANCE.md §2.3 update, including the five-vs-six discrepancy

Never resolved across the entire program: WO#11's postmortem flagged
that WO#11's own text said both "six duplicate implementations" and
"five remaining" in adjacent sentences, and that its own HARD BOUNDARIES
named exactly six files/call-sites. No session has ever had GOVERNANCE.md's
real §2.3 text in hand to settle this. **A plausible reconciliation**,
offered again here since it's now more clearly supportable with the
series complete: `job_agents.py`, `recipe_agents.py`,
`weekly_agents.py`+`workout_plan_ai_generator.py`+`media_recommend.py`
(migrated together as one batch in WO#13), `blog_agents.py`'s three
providers (one file, three separate migrations), and `finance_upload.py`
(intentionally left unmigrated) — **but this is still an inference**,
not a confirmed reading of the governance document. Whoever next has
access to the real `GOVERNANCE.md` should settle it directly rather than
propagate this document's guess into a seventh postmortem.

### G.5 — Remaining `blog_agents.py` items (Track B, small, not urgent)

1. **Stray leftover comment**: `# Add this constant near the top with
   the other model IDs`, directly above `_CEREBRAS_INTER_REQUEST_SLEEP` —
   reads like an unresolved artifact from an earlier incomplete edit
   (the constant isn't near the other model IDs; those live in
   `services/ai/providers/*.py` now). Not touched by WO#16 — not in its
   Task 2 list.
2. **`_CEREBRAS_INTER_REQUEST_SLEEP`'s disposition** — flagged by WO#15,
   left in place by WO#16 pending confirmation with whoever owns
   `life_os_code_improve.py` (which has its own separate
   `INTER_REQUEST_DELAY_SEC` constant, suggesting this one may be
   vestigial — but "may be" is not "confirmed").
3. **Duplicate `_estimate_tokens()` definition** — pre-existing,
   confirmed present (twice) as of WO#16, untouched by every work order
   in this chain by explicit policy. Needs its own standalone ticket
   per GOVERNANCE §4.5, not a silent fold-in.
4. **The commented-out prior `_cerebras()` implementation block** — both
   the original WO#15 text and WO#19's own drafted instructions assume
   this block exists (using it and the `# ── CEREBRAS MODEL IDs ──`
   header as landmarks). **Three consecutive sessions now (WO#14's
   postmortem §9.8, WO#15's postmortem §4.5, and this document, since
   WO#16 also never saw it) have been unable to confirm whether it
   actually exists in the real repository** — it simply wasn't present
   in any copy of `blog_agents.py` any session in this program was
   given. If it does exist in the real file, WO#19's landmark-based
   removal instructions are already broken (the live `_cerebras()`
   function and the `# ── CEREBRAS MODEL IDs ──` header WO#19 planned to
   use as anchors were both removed by WO#15) and need rewriting against
   a landmark that survives, per the `work_order_14-20_amendments.md`
   file's own fallback guidance for this exact situation.
5. **The correction to WO#14/#15's own "definition of done" grep
   checklist** — see Part C above. `_summarize_recent_difficulties()`
   and `_validate_difficulty_distribution()` are legitimate, in-scope,
   non-provider helpers; the checklist should read "...returns only
   `_detect_file_type`, the duplicate `_estimate_tokens`,
   `_summarize_recent_difficulties`, and
   `_validate_difficulty_distribution` — the latter two are non-provider
   helpers, out of scope by design, not leftover cleanup."

### G.6 — A stale-documentation finding surfaced while assembling this closeout (Track A, worth flagging before planning further work)

**GOVERNANCE.md §3.3** ("Migration Debt Tracker") and the **Master
Index** document both describe `finance`, `journal`, `recipes`+`pantry`,
`workout`, `media`, and `planning` as **not yet migrated** into the
`domains/` structure — the Master Index specifically marks WO#5 through
WO#10 as "📝 Drafted, not yet executed."

**This does not match the rest of the source material available across
this program.** Real, working file paths under `domains/finance/routers/`,
`domains/workout/routers/`, `domains/media/routers/`, and
`domains/recipes/` (implied by import paths in `recipe_agents.py`'s
callers) are referenced and edited throughout WO#13, #14, #15, and #16
as already-existing, current locations — not proposed future ones. The
`work_order_14-20_amendments.md` file goes further and cites specific,
detailed **postmortem section numbers** for WO#10 (§4.2, §4.3, §4.4,
§4.7, §4.11) as if that work order's execution and postmortem already
exist and were consulted directly.

**This is exactly the kind of drift GOVERNANCE.md's own §6 (Amendment
Process) exists to catch, and it hasn't been caught.** Two documents
that are supposed to be the authoritative status tracker for the Domain
Migration track (GOVERNANCE §3.3 and the Master Index) are stale
relative to what the rest of the program's own documents treat as
settled fact. **Recommendation, not an action taken here:** before
scoping any further Domain Migration work (or trusting this document's
own Track A statements above), confirm directly against the real
repository which of WO#5 through WO#10 have actually run, and update
GOVERNANCE.md §3.3 and the Master Index to match — don't let a future
session inherit two contradictory "ground truth" documents without at
least one of them being flagged as outdated.

### G.7 — Other tracks, status pointers only

- **Track C (WO#17, frontend consolidation):** independent of A/B, not
  touched by anything in this document's scope. Status per the amendments
  file: several template paths were corrected once WO#2/#3/#7/#10
  were confirmed to have run — see G.6's caveat about trusting that
  confirmation.
- **Track D (WO#18, DAG reorg):** independent. Two DAGs
  (`life_os_staging_promoter.py`, `life_os_refresh_streaming_availability.py`)
  were added to its scope after the original draft, per the amendments file.
- **Track E (WO#19, misc cleanup):** its Task 1 (dead-code removal in
  `blog_agents.py`) has an ordering dependency on whether WO#15 has run
  — it has, so Task 1's original landmark-based instructions are already
  stale per G.5, item 4, above.

### G.8 — Full-Program "Everything Is Actually Done" Sign-Off Checklist

Not considered complete until every box below is checked or explicitly,
individually re-deferred with a stated reason — mirroring the discipline
GOVERNANCE.md §4.6 applies to a single domain migration, extended to the
whole program:

- [ ] G.1: real `services/ai/base.py`/`keys.py`/`__init__.py` swapped in,
      every payload-shape and retry-path check across WO#11–16 re-run
      against them.
- [ ] G.2: `models.py` question resolved with fresh, direct reads (not
      this document's snapshot) if and when a domain's `models.py` is
      actually needed for something — not pre-emptively "fixed" absent
      a real finding.
- [ ] G.3: every unchecked box in the Track B Definition of Done resolved.
- [ ] G.4: GOVERNANCE.md §2.3 read directly, five-vs-six discrepancy settled.
- [ ] G.5: `blog_agents.py`'s five remaining small items each closed or
      explicitly re-deferred with a reason, individually.
- [ ] G.6: Domain Migration track's actual execution status confirmed
      against the real repository; GOVERNANCE.md §3.3 and the Master
      Index updated to match, or explicitly flagged as intentionally
      left stale with a reason.
- [ ] Track A (WO#1–10, #20), Track C (WO#17), Track D (WO#18), Track E
      (WO#19) each independently confirmed complete per their own
      postmortems' acceptance criteria — not assumed complete because
      other documents refer to them as done (see G.6).
- [ ] A single, current person or session has read this entire document
      and can state, in one sentence per track, what's actually left —
      not inferred from six different postmortems with six different
      snapshots of "current" state.
