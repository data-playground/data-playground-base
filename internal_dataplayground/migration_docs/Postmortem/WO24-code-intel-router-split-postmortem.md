# Work Order #24 — Post-Mortem: Code Intel Router Split

**Status:** Complete, with one open item requiring reviewer sign-off (see §4).
**Domain:** `domains/code_intel/`
**Track:** Track C (router-size splits, per WO#19's lint script — runs in
parallel with WO#23 and any other Track A/B/C/D work order; no shared
files with any of them).
**Companion documents:** `work_order_24_code_intel_router_split.md` (the
originating work order), `GOVERNANCE.md` §1.2 (file size limits) and §4
(AI Collaboration Guidelines — this postmortem follows the §4.3 template's
Output Format and the §4.4 review checklist).

---

## 1. Overview

WO#24 split three oversized Code Intelligence routers —
`ci_readme.py` (448 lines), `ci_files.py` (382 lines), and
`ci_projects.py` (328 lines) — into six files, all but one now under the
300-line ceiling from `CONTRIBUTING.md` / `GOVERNANCE.md` §1.2. This was a
pure location-and-organization refactor: no endpoint's URL, HTTP method,
request/response shape, or template name changed, and no model or schema
code was touched.

This document is the permanent record of **what WO#24 actually did**
(§2), **what was found that the original work order didn't fully resolve
and what's proposed to close it out** (§3–4), and **what remains as
deferred, cross-cutting cleanup that should happen once every Track C
work order — not just this one — has landed** (§5). Section 5 is written
so a future work-order agent can pick it up without re-deriving context.

---

## 2. What Was Done (Initial Work Order, As Executed)

### 2.1 Part A — `ci_readme.py` (448 → 287 lines)
- Moved the folder-README lookup helpers (`_get_folder_readme`,
  `_get_latest_folder_readme`) and the two folder-README endpoints
  (`PATCH`/`GET .../folder-readme`) into a new
  `domains/code_intel/routers/ci_folder_readme.py` (195 lines).
- `ci_readme.py` retained `generate_readme`, `save_readme_edits`,
  `push_readme`, `trigger_readme_dag` — unmodified except for the import
  swap below.
- Updated `ci_projects.py`'s import of `_get_folder_readme` /
  `_get_latest_folder_readme` to point at `ci_folder_readme` instead of
  `ci_readme`.
- `ci_readme.py` itself also now imports `_get_latest_folder_readme` from
  `ci_folder_readme` (it's used in four of its four remaining endpoints
  to keep the folder-README panel populated on re-render).
- **Result: fully resolved.** Both files landed under 300 lines with no
  open items.

### 2.2 Part B — `ci_files.py` (382 → 324 lines)
- Moved the three batch-Airflow-trigger endpoints
  (`trigger_batch_narrate`, `trigger_batch_comment`,
  `trigger_batch_improve`) and the `CODE_*_DAG` identifier constants into
  a new `domains/code_intel/routers/ci_batch.py` (84 lines).
- Removed the now-unused `trigger_airflow` import from `ci_files.py`.
- Per the work order's explicit instruction, the seven inline
  single-file endpoints (`file_detail`, `pull_file`, `narrate_file`,
  `comment_file`, `improve_file`, `push_commented_file`,
  `update_comment_status`) and **all their existing docstrings** were
  left untouched in `ci_files.py`.
- **Result: not fully resolved.** See §3 — `ci_files.py` is still 324
  lines, over the 300-line ceiling. This is the one open item from the
  initial work order.

### 2.3 Part C — `ci_projects.py` (328 → 216 lines)
- Per the work order's conditional instruction, line count was
  re-checked *after* Parts A and B landed (328 lines — still over 300, so
  the split proceeded rather than being skipped).
- Moved the two polling/badge-refresh endpoints (`get_file_statuses`,
  `project_status`) into a new `domains/code_intel/routers/ci_status.py`
  (138 lines).
- Dropped the now-unused `_get_folder_readme`, `text`, and `JSONResponse`
  imports from `ci_projects.py` (only `project_status`, which moved out,
  used them); kept `_get_latest_folder_readme`, still used by
  `sync_files_from_github` and `project_detail`.
- **Result: fully resolved.** Both files landed under 300 lines with no
  open items.

### 2.4 `main.py`
- Import line updated to pull in `ci_folder_readme`, `ci_batch`,
  `ci_status` alongside the original three modules.
- Each new router was registered immediately after the file it split
  from, preserving the original "files + readme before projects" relative
  ordering the pre-existing comment called out:
  `ci_files → ci_batch → ci_readme → ci_folder_readme → ci_projects → ci_status`.
- The registration comment was expanded to explain *why* each new router
  sits where it does, so the rationale doesn't silently rot the next time
  someone edits this block.

### 2.5 Verification performed
- **Line counts**, before and after, for all six files (see §2.1–2.3
  above and the table in §6).
- **Endpoint inventory**: grepped every `@router.<method>(...)` decorator
  across all six files and confirmed all 23 original endpoints are
  present exactly once, with zero path/method collisions. Full list in
  §6.
- **Syntax check**: `ast.parse()` on all six router files and `main.py`.
  (This confirms syntactic validity only — these files were not run
  against a live FastAPI app or database in this session, since no
  server, DB, or GitHub/Airflow credentials were available. A human or a
  follow-up agent with repo + service access should still do a live
  smoke test of each endpoint before merge.)
- **Substitution noted**: `scripts/check_router_line_limits.py` (the
  canonical lint script the work order asks to run first) was not
  accessible in this session — no repo checkout, no network access.
  `wc -l` was used as the closest achievable substitute to confirm the
  three files were genuinely over 300 lines before splitting, and to
  verify the results after. This substitution should be closed out per
  §5.4 before the work order is marked fully verified.

---

## 3. What Was Found During Execution (Deviation From the Plan)

The work order's Part B steps are explicit and specific: move only the
three batch endpoints, and **keep** the seven inline endpoints together
with their docstrings in `ci_files.py` ("these carry the heaviest
docstrings, which is most of why this file is long; keep the docstrings,
they're doing real documentation work per this codebase's conventions").

Executed literally, this leaves `ci_files.py` at 324 lines — over the
300-line ceiling the same work order's acceptance criteria require ("All
resulting files under 300 lines"). This is a direct conflict between two
parts of the same work order, not a pre-existing bug and not a mistake in
execution: the instructions as written cannot simultaneously satisfy both
constraints.

Per the work order template's own HARD BOUNDARIES ("If instructions
conflict with actual code found, STOP and report — don't improvise"),
execution stopped at that boundary rather than unilaterally trimming a
docstring or moving an eighth endpoint that wasn't authorized. Parts A and
C, which had no such conflict, were completed in full.

---

## 4. What's Proposed To Close Out the Open Item (Pending Reviewer Sign-Off)

**No further code change has been made to `ci_files.py` beyond what's
described in §2.2.** The two options below are proposed for the reviewer
to choose between — this postmortem does not treat either as already
agreed, since that decision hasn't been made yet:

**Option A — Accept 324 lines as a documented exception.**
`ci_files.py` stays as-is. The exception gets recorded (in
`GOVERNANCE.md` §1.2, alongside the other file-size rules) with a one-line
rationale: this file's length is almost entirely earned by Google-style
docstrings on `improve_file`/etc. that the codebase's own conventions
(§1.3) require, and splitting further would either duplicate those
docstrings across files or thin them out — both worse than a 24-line
overage.

**Option B — Move `update_comment_status` (the smallest, most
self-contained of the seven remaining endpoints, ~30 lines, no docstring)
into a follow-up file** (e.g. `ci_comment_status.py`, or fold it into
`ci_batch.py` since it's also a small "status transition" endpoint — a
naming/grouping decision the reviewer should make, not this document).
This would bring `ci_files.py` to roughly 294 lines, clearing the ceiling
without touching any of the six endpoints/docstrings the original work
order explicitly asked to preserve as-is.

**Recommendation:** Option B keeps the codebase's stated rule ("no router
file exceeds 300 lines, hard ceiling") uniformly true with no documented
exceptions to track, and costs one small, low-risk file move. Whichever
option the reviewer picks, this section should be updated to say which
was chosen and why, once decided — that update is itself the last step
of closing out WO#24.

---

## 5. Post-Migration Follow-Up (To Be Done After All Related Migrations Complete)

This section is the one requested for a future agent to coalesce
requirements from. It's split into (a) items this specific work order's
diff does **not** touch and should **not** be assumed to have handled,
and (b) genuine cross-cutting cleanup that only makes sense once every
Track C (router-split) work order — not just WO#24 — has landed.

### 5.1 `models.py` — explicitly out of scope for this work order

**No `models.py` file, at any level, was read, touched, or needs updating
as a result of WO#24.** This was a pure router-layer (HTTP
endpoint/controller) split — no ORM class, enum, or schema was moved,
renamed, or removed, so there are no stale model references to clean up
from this work order specifically. Recording this explicitly so a future
agent doesn't assume router-split work orders imply model-layer cleanup
by default:

- **Root `models.py`**: per `GOVERNANCE.md` §2.4, this file's shim
  mechanism is already **historical/closed** — WO#20 removed every
  remaining domain shim, and WO#22 removed root `models.py` entirely once
  a repo-wide grep confirmed it had no real consumers left. There is
  nothing left in that file to adjust, for this domain or any other.
- **`domains/code_intel/models.py`**: not part of this work order's
  SCOPE, not opened during execution, and nothing in `ci_readme.py`,
  `ci_files.py`, or `ci_projects.py`'s original content suggested it
  contains router-specific references that a router split would
  invalidate (routers import *from* models, never the reverse, which is
  the expected direction and isn't touched by moving route handlers
  between files).
- **If a future work order does need to touch
  `domains/code_intel/models.py`** (e.g. to relocate the `ReadmeStatus`,
  `CommentedStatus`, `ImprovementStatus`, or `FolderReadmeStatus` enums,
  or to change the `CodeProject.files` / `CodeFile.blog_ideas` cross-domain
  relationships), that is a **separate, model-layer change** and should
  get its own work order and its own postmortem per `GOVERNANCE.md` §4.5
  ("Bugs Found During Migration Are Not Migration Work" — the same
  separation-of-concerns logic applies to unrelated model changes
  surfacing here). Do not bundle it into a router-split cleanup pass.

### 5.2 `GOVERNANCE.md` §1.2 is now stale for this domain

§1.2 currently reads: *"`weekly_plan.py`, `media_recommend.py`,
`workout_plans.py`, and `ci_readme.py` all exceeded it at time of
writing."* `ci_readme.py` no longer belongs on that list (287 lines as of
this postmortem). This document (WO#24's diff) does **not** edit
`GOVERNANCE.md` — that file wasn't in WO#24's SCOPE, and editing it
mid-work-order would blur "pure relocation" review the same way bundling
a bug fix would (§4.5's logic again).

**Action for after all Track C work orders complete:** do one pass over
§1.2 that:
1. Removes `ci_readme.py` from the "exceeded it at time of writing" list
   (or moves the whole sentence to past tense / a "historical" framing,
   mirroring exactly how §2.4 was rewritten to "Status:
   historical/closed" after WO#20/WO#22 resolved it).
2. Checks whether `weekly_plan.py`, `media_recommend.py`, and
   `workout_plans.py` have also been resolved by their own Track C work
   orders by that point, and updates or removes them from the list
   accordingly, file by file (don't assume — verify each).
3. Confirms `habits.py` (WO#23) is reflected accurately too — it's not
   currently named in §1.2's example list, but if WO#23 has landed by
   then, check whether it should be added to the *resolved* precedent
   list (alongside the `workout_plans.py` → `_shared.py` split pattern
   §1.2 already cites as precedent).

This should land as **one repo-wide edit once every Track C work order is
verified**, not once per work order — doing it per-work-order would mean
`GOVERNANCE.md` gets N small edits instead of one coherent pass, and risks
a later work order's edit clobbering an earlier one's.

### 5.3 Wire up the CI-checkable lint step §1.2 asks for

§1.2 states the 300-line ceiling "should be a CI-checkable lint step..,
not just a reviewer's judgment call" but, as of WO#24, this is still
manual (`scripts/check_router_line_limits.py` exists and was referenced
by both WO#23 and WO#24, but nothing in the provided `main.py` or repo
docs indicates it's wired into CI yet).

**Action for after all Track C work orders complete:** once every
oversized router identified by the original lint pass has a landed split
(or a documented exception per §4's Option A pattern, if any work order
ends up choosing that route), add `scripts/check_router_line_limits.py`
as an actual CI gate (pre-commit hook or CI pipeline step) so file-size
regressions are caught automatically going forward, rather than requiring
another manual lint-script sweep next time a router grows past 300 lines.

### 5.4 Close out the lint-script substitution noted in §2.5

This postmortem's line-count verification used `wc -l` as a stand-in for
`scripts/check_router_line_limits.py`, which wasn't accessible in this
session. Once a follow-up agent has real repo access, it should re-run
the actual script against all six Code Intel router files (and ideally
the full repo) to confirm parity with the `wc -l` figures in §6 before
this work order is considered independently verified. If the real script
counts lines differently (e.g. excluding blank lines or docstrings), the
§4 decision about `ci_files.py` may need revisiting.

### 5.5 `main.py` ordering comment — candidate for centralization

WO#24 expanded the inline comment above the Code Intel router
registrations to explain the new six-router ordering (§2.4). Once all
Track C splits have landed across all domains, it may be worth asking
whether these per-domain "registration order matters because of path
specificity" explanations (Code Intel's is one of several similar
comments already in `main.py`, e.g. the `/static` mount-ordering note)
should be consolidated into a single `GOVERNANCE.md` section (candidate:
extending §2.6 "Templating & Static Serving" into a broader "Registration
Ordering" section) rather than living as scattered inline comments that
each future split has to notice and preserve by hand. Not urgent — flagged
for the same kind of "worth generalizing" treatment §6 (Amendment Process)
already describes for other one-off corrections.

---

## 6. Reviewer Sign-Off Checklist

| # | Item | Status |
|---|------|--------|
| 1 | `ci_readme.py` under 300 lines (287) | ✅ |
| 2 | `ci_folder_readme.py` created, under 300 lines (195) | ✅ |
| 3 | `ci_batch.py` created, under 300 lines (84) | ✅ |
| 4 | `ci_files.py` under 300 lines | ⚠️ 324 lines — see §3–4, needs reviewer decision (Option A or B) |
| 5 | `ci_projects.py` under 300 lines (216) | ✅ |
| 6 | `ci_status.py` created, under 300 lines (138) | ✅ |
| 7 | All 23 original endpoints present, zero path/method collisions | ✅ (see inventory below) |
| 8 | `main.py` imports + registers all six routers, ordering preserved | ✅ |
| 9 | No endpoint URL, method, request/response shape, or template name changed | ✅ |
| 10 | No model/schema file touched | ✅ (see §5.1) |
| 11 | All files pass `ast.parse()` syntax check | ✅ |
| 12 | Verified against real `scripts/check_router_line_limits.py` (not just `wc -l`) | ⚠️ Pending — see §5.4 |
| 13 | Live smoke test of all 23 endpoints against a running app + DB | ⚠️ Pending — not possible in this session, see §2.5 |
| 14 | `GOVERNANCE.md` §1.2 updated to drop `ci_readme.py` from the stale list | ⚠️ Deferred by design — see §5.2 (batch with other Track C work orders) |
| 15 | CI lint gate for the 300-line rule wired up | ⚠️ Deferred by design — see §5.3 |

**Endpoint inventory (method + path, all under prefix `/code-intel`):**

| File | Endpoints |
|---|---|
| `ci_files.py` | `GET /files/{id}`, `POST /files/{id}/pull`, `POST /files/{id}/narrate`, `POST /files/{id}/comment`, `POST /files/{id}/improve`, `POST /files/{id}/push-comments`, `PATCH /files/{id}/comment-status` |
| `ci_batch.py` | `POST /batch/narrate`, `POST /batch/comment`, `POST /batch/improve` |
| `ci_readme.py` | `POST /projects/{id}/generate-readme`, `PATCH /projects/{id}/readme`, `POST /projects/{id}/push-readme`, `POST /projects/{id}/trigger-readme` |
| `ci_folder_readme.py` | `PATCH /projects/{id}/folder-readme`, `GET /projects/{id}/folder-readme` |
| `ci_projects.py` | `GET ` (i.e. `/code-intel`), `POST /projects`, `DELETE /projects/{id}`, `POST /projects/{id}/sync`, `GET /projects/{id}/detail` |
| `ci_status.py` | `GET /projects/{id}/file-statuses`, `GET /projects/{id}/status` |

**Line count summary:**

| File | Before | After |
|---|---|---|
| `ci_readme.py` | 448 | 287 |
| `ci_files.py` | 382 | 324 |
| `ci_projects.py` | 328 | 216 |
| `ci_folder_readme.py` | — (new) | 195 |
| `ci_batch.py` | — (new) | 84 |
| `ci_status.py` | — (new) | 138 |
| **Total** | **1158** | **1244** |

(The total line count grew, as expected for any file split — each new
file carries its own header docstring, imports, and section-divider
comments, which duplicates a small amount of boilerplate that a single
file didn't need to repeat.)

---

## 7. Rollback

`git checkout` on `domains/code_intel/routers/ci_files.py`,
`ci_readme.py`, `ci_projects.py`, and `main.py`; delete (or `git rm`)
`ci_folder_readme.py`, `ci_batch.py`, and `ci_status.py`. No database
migration, model change, or data backfill is involved, so rollback is a
pure file-level revert with no other cleanup required.
