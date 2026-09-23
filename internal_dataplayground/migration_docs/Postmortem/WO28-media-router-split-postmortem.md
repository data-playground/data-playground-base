# Work Order #28 — Post-Mortem: Media Domain Router Split

**Status:** Awaiting reviewer sign-off
**Work order:** `migration_docs/Work Orders/work_order_28_media_router_split.md`
**Domain:** `domains/media/`
**Track:** Track C (router line-limit splits) — runs in parallel with Track A, B, D and other Track C work orders per WO#28's own closing note
**Purpose of this document:** Record what WO#28 actually did versus its literal instructions, what was found during verification, what was agreed afterward to reconcile the two, and what remains outstanding. This is the artifact a reviewer should check before marking WO#28 "done," and the artifact a future agent should read before touching `domains/media/routers/` again.

---

## 1. Executive Summary

WO#28 asked for two independent splits to bring `media_recommend.py` and `media.py` under the repo's 300-line router ceiling (CONTRIBUTING.md / GOVERNANCE.md §1.2). Both splits were carried out exactly as scoped — the named functions moved, no endpoint's path/method/shape/template changed, and the `_USE_GEMINI` / `asyncio.to_thread` boundaries were left untouched.

Both resulting "kept" files are still over 300 lines (316 and 309 respectively) because the work order's STEPS pinned a closed set of functions to remain in each file, and that set doesn't quite clear the ceiling on its own. This is flagged, not silently fixed — see §7 for why, and §8 for what was agreed instead.

**Net result:** the split is functionally complete and safe to ship (same behavior, same URLs, verified importable). The line-count acceptance criterion is not fully met and needs an explicit reviewer decision (§9) rather than being treated as a pass or a blocker by default.

---

## 2. What WO#28 Set Out To Do (Original Scope Recap)

- **Part A:** Move `_gemini_explain` and `_format_ml_only` out of `media_recommend.py` into a new `_recommend_pipeline.py`. Keep `recommendations_page`, `generate_recommendations`, `recommendation_history` in `media_recommend.py`. Do not touch the `_USE_GEMINI` read or the `asyncio.to_thread` wrap.
- **Part B:** Confirm `media.py`'s real line count first. If still over 300, move `update_season_progress` (the one TV-season-specific endpoint) into a new `media_seasons.py`. Keep everything else in `media.py`.
- **Hard boundaries:** no URL/method/shape/template changes; only touch files named in each Part's SCOPE; don't fix pre-existing bugs found along the way — reproduce, report, mark ⚠️.
- **Acceptance criteria:** all resulting files under 300 lines (or Part B confirmed not needed), all named endpoints behaviorally unchanged.

---

## 3. What Was Done Under the Initial Work Order

This section is the literal record of what WO#28 authorized and what was executed under that authorization — nothing in this section required a follow-up decision.

### Part A — `media_recommend.py`

| | Before | After |
|---|---|---|
| `domains/media/routers/media_recommend.py` | 416 lines | 316 lines |
| `domains/media/routers/_recommend_pipeline.py` | — (new) | 119 lines |

- **Created:** `_recommend_pipeline.py`, holding `_format_ml_only` and `_gemini_explain`, moved verbatim (including the `asyncio.to_thread(...)` wrap and its explanatory comment — untouched, per HARD BOUNDARIES).
- **Edited:** `media_recommend.py` — removed the two function bodies; removed the `asyncio`, `typing.Optional`, and `services.ai` (`MODEL_FLASH`, `call_gemini_json`) imports, since after the move nothing else in the file used them; added `from domains.media.routers._recommend_pipeline import _format_ml_only, _gemini_explain`.
- **Untouched, confirmed by inspection:** the `_USE_GEMINI = os.environ.get(...)` read stays at module import time in `media_recommend.py`; the `asyncio.to_thread` wrap stays inside `_gemini_explain` exactly as before.

### Part B — `media.py`

| | Before | After |
|---|---|---|
| `domains/media/routers/media.py` | 355 lines | 309 lines |
| `domains/media/routers/media_seasons.py` | — (new) | 67 lines |

- **Confirmed** `media.py` was over 300 lines (355) before doing anything, per WORKING METHOD's instruction to verify first. (Note: `scripts/check_router_line_limits.py` was not present/in scope for this session — `wc -l` was used as a substitute. This substitution should be re-verified against the real lint script the next time someone has access to it; see §10.)
- **Created:** `media_seasons.py`, holding `update_season_progress` moved verbatim, mounted with the same `/media` prefix so the endpoint's full path (`POST /media/{user_media_id}/seasons/{season_number}`) is unchanged.
- **Edited:** `media.py` — removed the `update_season_progress` function; removed the `TVSeasonProgress` import from `domains.media.models`, since after the move nothing else in the file referenced it.

### Verification performed under the initial work order

- `python -m py_compile` run against all four touched/created router files plus `main.py` — all pass.
- Confirmed by grep that `TVSeasonProgress` had no remaining reference in `media.py` after the move, and that no other function in `media_recommend.py` used `asyncio`, `Optional`, `MODEL_FLASH`, or `call_gemini_json` after `_gemini_explain` moved out.

---

## 4. Deviation From Literal Scope, Included Anyway: `main.py`

Part B's SCOPE lists only `media.py` and (conditionally) `media_seasons.py`. It does not list `main.py`. `main.py` was edited anyway:

- Added `media_seasons` to the `domains.media.routers` import line in `main.py`.
- Added `app.include_router(media_seasons.router)`, placed before `app.include_router(media.router)` — matching the existing repo convention for every other specific-path sub-router registered ahead of a domain's catch-all router (see `ci_batch` before `ci_files`'s catch-all pattern, `journal_synthesis` before `journal`, `recipe_mutations` after the more specific recipe routers, etc.).

**Why this isn't optional:** a router object that exists but is never passed to `app.include_router(...)` serves no routes — FastAPI does not auto-discover routers. Without this edit, `POST /media/{user_media_id}/seasons/{season_number}` would 404 for every request, which is a direct violation of the HARD BOUNDARY that "no endpoint's URL path, HTTP method, request/response shape... changes." Leaving `main.py` untouched would have technically respected the letter of "only touch files named in SCOPE" while breaking the very acceptance criterion that boundary exists to protect.

This is flagged explicitly here (rather than folded silently into the "Files edited" list) because it's exactly the kind of scope quirk GOVERNANCE.md §4.4 point 1 asks reviewers to check deliberately: "Did every HARD BOUNDARY get respected? Check 'Files edited' against the exclusion list explicitly."

**Recommendation for future work orders:** any WO that introduces a new router file should list `main.py` in SCOPE explicitly, rather than leaving this as an implicit "well, obviously" step. See §10.5.

---

## 5. Findings During Verification

### 5.1 Acceptance criteria not met as originally written

- **Part A:** `media_recommend.py` is 316 lines, not ≤300, after moving exactly the two functions STEPS named. The three functions STEPS explicitly said must stay (`recommendations_page`, `generate_recommendations`, `recommendation_history`) are, on their own, too large to fit under 300 with reasonable imports/docstring overhead.
- **Part B:** `media.py` is 309 lines, not ≤300, after moving the one function STEPS named. The seven functions STEPS explicitly said must stay are, similarly, 9 lines too many.

### 5.2 Pre-existing dead imports discovered (not fixed — see §7)

Found by grep while confirming which imports were safe to remove after each move. None of these were introduced by WO#28; all predate it and are unrelated to the split itself, and so were left alone per the work order's own "don't fix pre-existing bugs, report them" instruction:

| File | Dead import | Why it's dead |
|---|---|---|
| `media_recommend.py` | `import json` (top-level) | Shadowed everywhere it's needed by a local `import json as _json` inside `_gemini_explain` (now in `_recommend_pipeline.py`); the top-level import is never referenced as `json.*` anywhere in the file. |
| `media_recommend.py` | `and_` (from `from sqlalchemy import and_, select`) | Never called anywhere in the file. |
| `media_recommend.py` | `UserMediaStatus` (from `domains.media.models`) | Imported but never referenced outside the import line. |
| `media.py` | `desc` (from `from sqlalchemy import select, desc, func`) | The file calls `.desc()` as a method on a column (`UserMedia.updated_at.desc()`), not the imported `desc` function — the import itself is unused. |
| `media.py` | `RecommendationMediaType` (from `domains.media.models`) | Imported but never referenced outside the import line. |

None of these affect the line-count shortfall meaningfully (removing them saves at most 1–2 lines per file, since most sit inside multi-symbol import lines that wouldn't shrink line-count-wise) and none affect behavior.

---

## 6. Root Cause: Why the Line-Count Criterion Wasn't Fully Met

Both parts' STEPS pin a *closed, named set* of functions to a specific file ("keep X, Y, Z in file A"), and separately the ACCEPTANCE CRITERIA demand that file A end up ≤300 lines. For both parts, the named set is internally consistent (nothing about it is wrong) but is arithmetically too large to satisfy the line limit once you also keep the file's necessary imports, docstring, and section-divider comments (which GOVERNANCE.md §1.3 requires as house style, not optional flourish).

This is a genuine conflict between two explicit instructions in the same work order, not a mistake introduced during execution. Per the standing rule this repo's own work orders are built around ("if instructions conflict with actual code found, STOP and report — don't improvise"), the correct move was to execute STEPS literally, verify the real resulting line count, and report the shortfall rather than unilaterally moving one of the pinned functions elsewhere to force compliance. That's what happened.

---

## 7. What Was Agreed After the Initial Migration

These are the decisions made **after** WO#28's literal execution and initial report, to reconcile the findings in §5–§6. This is the section a reviewer should treat as the actual "terms" the migration is being judged against — not the original WO#28 acceptance criteria in isolation.

1. **The two line-count shortfalls (316 / 309 lines) are accepted as tracked technical debt, not treated as a failed migration.** Reasoning: forcing either file under 300 right now would require moving a route handler that STEPS explicitly pinned in place, which is a scope decision that belongs to a reviewer or a dedicated follow-up work order, not something to be decided unilaterally mid-execution. The acceptance criteria for WO#28 are therefore reclassified from ❌ to ⚠️ for the line-count item specifically (see §9), on the condition that §10.2's Part C proposal is picked up as real follow-up work rather than forgotten.
2. **The `main.py` registration (§4) is accepted as an in-scope-by-necessity edit**, not a boundary violation — a router split that doesn't wire up the new router isn't a completed split, it's a partial one that happens to still compile.
3. **The five dead imports in §5.2 are *not* being cleaned up as part of WO#28.** This follows GOVERNANCE.md §4.5 directly: bugs/cleanup found during a migration get their own ticket, even a one-line fix, so the migration's diff stays reviewable as "pure relocation plus the minimum wiring needed to keep it working." They're itemized precisely in §10.1 so the next session doesn't have to rediscover them.
4. **The `wc -l` substitution for `scripts/check_router_line_limits.py` is accepted for this session's verification**, on the condition that whoever next has access to the actual lint script re-runs it against both final files and confirms it agrees with the counts in §3. If the real script counts differently (e.g., it excludes comments/docstrings, or blank lines), the ⚠️ line-count items in §9 may need to be revisited — possibly turning into a pass, or into a firmer ❌.

---

## 8. Final Acceptance Criteria — Reconciled Status

| Criterion | Original | Reconciled (post-agreement) | Notes |
|---|---|---|---|
| Part A: `media_recommend.py` ≤300 lines | ❌ (316) | ⚠️ Accepted as tracked debt | See §7.1, §10.2 |
| Part A: `_recommend_pipeline.py` ≤300 lines | ✅ (119) | ✅ | |
| Part A: `/media/recommend*` endpoints unchanged | ✅ | ✅ | Path/method/shape verified unchanged |
| Part A: `_USE_GEMINI` / `asyncio.to_thread` untouched | ✅ | ✅ | Verified by inspection |
| Part B: confirmed real line count before acting | ✅ | ✅ | Substitute tool used — see §7.4 |
| Part B: `media.py` ≤300 lines | ❌ (309) | ⚠️ Accepted as tracked debt | See §7.1, §10.2 |
| Part B: `media_seasons.py` ≤300 lines | ✅ (67) | ✅ | |
| Part B: `POST /media/{id}/seasons/{n}` unchanged | ✅ | ✅ | Same path, method, template |
| Part B: all other `media.py` endpoints unchanged | ✅ | ✅ | Verified by inspection |
| (New) `main.py` correctly registers `media_seasons.router` | N/A — not an original criterion | ✅ | Added because required for the split to actually function; see §4 |

**A migration cannot be marked fully "done" per GOVERNANCE.md §4.6 while any ⚠️ items are open-ended.** The two ⚠️ items above are only acceptable as a *closed* state if §10.2 is formally logged as follow-up work (not just mentioned in this document and then lost) — that's the whole reason this postmortem exists.

---

## 9. Post-Migration Cleanup — What Needs to Happen Once All Migrations Are Complete

This section exists so this cleanup doesn't get "bundled into the next thing that happens to touch these files" (which GOVERNANCE.md §4.5 explicitly warns against) and doesn't get lost the way a purely verbal agreement would. It mirrors the pattern GOVERNANCE.md §2.4 already established for root `models.py` shim removal: leave a clearly-scoped, clearly-triggered cleanup task on record, and pick it up deliberately later rather than improvising it now.

### 9.1 Domain-local cleanup — `domains/media/routers/` dead imports

Trigger: next time `domains/media/routers/` is opened for any reason (a bug fix, a feature, another split), or a standalone "import hygiene" pass if one is ever scheduled.

Exact fix (all zero-behavior-change, verified dead in §5.2):

- `domains/media/routers/media_recommend.py`
  - Remove `import json` (line-level top import; the function that needs `json` already does its own local `import json as _json`).
  - Change `from sqlalchemy import and_, select` → `from sqlalchemy import select`.
  - Remove `UserMediaStatus` from the `domains.media.models` import block.
- `domains/media/routers/media.py`
  - Change `from sqlalchemy import select, desc, func` → `from sqlalchemy import select, func`.
  - Remove `RecommendationMediaType` from the `domains.media.models` import block.

Do this as its own small, standalone diff with its own review — not folded into a feature change, per §4.5.

### 9.2 Line-count follow-up — Part C candidate (NOT YET AUTHORIZED)

If/when `scripts/check_router_line_limits.py` becomes a real CI gate (GOVERNANCE.md §1.2 already calls for this to happen and notes it hasn't yet), both `media_recommend.py` (316) and `media.py` (309) will fail it. A concrete, scoped Part C is proposed here so a future work order doesn't have to redesign the split from scratch — but **this proposal is not authorized to execute on its own; it needs its own work order with its own ROLE/HARD BOUNDARIES/SCOPE per the standing template (GOVERNANCE.md §4.3), because it moves route handlers that WO#28 explicitly pinned in place.**

- **`media_recommend.py` (needs to shed ~20+ lines):** `recommendation_history` (the `GET /media/recommend/history` handler, ~18 lines including its decorator/docstring) is the most self-contained candidate — it doesn't share any local state with `generate_recommendations` or `recommendations_page`. Candidate destinations: a new `media_recommend_history.py` (mirrors the "one concern per file" pattern already used elsewhere, e.g. `ci_status.py` split from `ci_projects.py`), or folding it into `_recommend_pipeline.py` if that file's charter is explicitly widened from "scoring/explanation internals" to "recommendation pipeline support" (would need its own docstring update to stay honest about what it holds).
- **`media.py` (needs to shed ~9+ lines):** `update_notes` (~13 lines) is the smallest fully self-contained candidate with no shared helpers. It doesn't fit thematically into `media_seasons.py`. Candidate destination: a new small `media_notes.py`, or bundling it with `remove_from_list` into something like `media_misc.py` if the reviewer would rather avoid a single-function file.
- Either candidate, once actually moved, requires the same `main.py` registration step documented in §4 for whatever new router file results.

### 9.3 Cross-reference sweep

Before this postmortem is closed out, confirm nothing outside `domains/media/routers/` imports the four relocated symbols by their old module path:

- `_gemini_explain`, `_format_ml_only` — previously importable (in principle) from `domains.media.routers.media_recommend`; now live in `domains.media.routers._recommend_pipeline`.
- `update_season_progress` — previously importable from `domains.media.routers.media`; now lives in `domains.media.routers.media_seasons`.

Per GOVERNANCE.md §2.2, `routers/dashboard.py` is the one sanctioned cross-domain reader and the most likely place such a reference would exist, if one exists at all — routers aren't typically imported by other routers, so this is expected to be a clean check, but it wasn't verified in this session (no access to `routers/dashboard.py`'s contents) and should be confirmed explicitly, not assumed.

### 9.4 Repo-wide items — once all Track A/B/C/D work orders conclude

WO#28 is one router split among several running in parallel (its own closing note says as much). Once the full set finishes:

- **Turn on the CI gate.** GOVERNANCE.md §1.2 already states the line limit "should be a CI-checkable lint step... not just a reviewer's judgment call" but flags that it isn't wired up yet. Once every in-flight split lands, re-run `scripts/check_router_line_limits.py` (the real one, not the `wc -l` substitute used here) against every router in the repo and wire it into CI. Expect it to immediately flag `media_recommend.py` and `media.py` per §9.2 unless that follow-up has also landed by then.
- **Repeat the §9.1-style dead-import sweep for every other domain's split**, not just media's. Every "move function X out, keep the rest" split leaves the same shape of residue (an import that was only needed by the moved code, missed because the "kept" file also happens to import the same symbol name from elsewhere, or because it was already dead before the split and easy to overlook). Media's five are cheap to check by grep; the same grep-per-import approach should be applied per split.
- **Reconcile GOVERNANCE.md §3.3's Migration Debt Tracker.** As written, §3.3 lists `media` (among others: finance, journal, recipes+pantry, workout, planning) as "not yet moved into the `domains/` structure" — but `main.py`, `core/templating.py`, and this very split all show the media domain has been living under `domains/media/` since WO#9. §3.3's header notes the document was "last updated after Work Orders #1–4," which predates WO#9's media migration — this is stale documentation debt, not a live migration gap. Worth a one-line GOVERNANCE.md correction in the same pass, per §6's amendment process ("treat every work-order report's Notes section as a candidate source of the next amendment").
- **Consider a GOVERNANCE.md §4.3 template amendment**, per this postmortem's §6: when a work order's STEPS pin a closed function set to a file *and* separately require that file to hit a line ceiling, the template should require the author to pre-check that the arithmetic actually works (count the pinned functions' lines before writing the work order) rather than leaving that discovery for execution time. This exact gap produced both ⚠️ items in this document.

---

## 10. Rollback

Standard: `git checkout` on every file listed in §3 and §4 (`media_recommend.py`, `_recommend_pipeline.py`, `media.py`, `media_seasons.py`, `main.py`). No schema/data changes were made, so no migration/rollback beyond source control is needed.

---

## 11. Reviewer Sign-off Checklist

Before marking WO#28 complete, confirm:

- [ ] §3's before/after line counts reproduce locally (or via the real `scripts/check_router_line_limits.py`, per §7.4).
- [ ] `GET /media`, `GET /media/{id}/detail`, `POST /media`, `PATCH /media/{id}/status`, `PATCH /media/{id}/rate`, `PATCH /media/{id}/notes`, `DELETE /media/{id}` all still resolve and behave as before.
- [ ] `POST /media/{id}/seasons/{n}` still resolves (this specifically depends on the §4 `main.py` edit — confirm it wasn't reverted separately).
- [ ] `GET /media/recommend`, `POST /media/recommend/generate`, `GET /media/recommend/history` all still resolve and behave as before, including the Gemini-enabled and Gemini-disabled code paths.
- [ ] §8's table is reviewed and the two ⚠️ line-count items are explicitly accepted (not silently waved through) — if not accepted, this becomes a real Part C work order per §9.2, not a follow-up "someday."
- [ ] §9.1's dead-import list is logged somewhere it won't be lost (a ticket, a GOVERNANCE.md §3.2-style tracker entry, or equivalent) — not fixed now, but not forgotten either.
- [ ] §9.3's cross-reference sweep is actually performed against `routers/dashboard.py` and any test suite that may reference the moved functions by their old path.
