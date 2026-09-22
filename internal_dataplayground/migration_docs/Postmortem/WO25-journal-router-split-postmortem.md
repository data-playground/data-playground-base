# WO#25 Postmortem — Journal Domain Router Split

**Work order:** #25 — `domains/journal/routers/journal.py` split
**Track:** C (per WO#25's "For the next work order" note — parallelizable
with every other Track C item, and with Tracks A, B, D)
**Status:** Complete, pending live-environment re-verification (see §4)
**Document date:** 2026-09-21

This document is the permanent record of what WO#25 actually did, what it
deliberately did *not* do, and — per the request that produced this
document — what still needs to happen **after all other domain
migrations are complete**. It exists so that (a) a reviewer can sign off
on WO#25 without confusing "explicitly out of scope" with "missed," and
(b) a future work order (or agent) picking up the deferred items has a
single, thorough source to coalesce requirements from, rather than
needing to re-derive them from the original WO#25 prompt and my
tool-verified findings.

Per GOVERNANCE.md §6 (Amendment Process), anything below that turns out
to generalize beyond journal/planning should eventually be promoted into
GOVERNANCE.md itself — that promotion is *not* done as part of this
document.

---

## 1. Why WO#25 existed

`domains/journal/routers/journal.py` was 446 lines — well over the
300-line hard ceiling in CONTRIBUTING.md and GOVERNANCE.md §1.2. This was
flagged as needing a pre-300 split, not a post-hoc fix. WO#25 was a pure
location-and-organization refactor: no endpoint path, method,
request/response shape, or template name was allowed to change, and no
schema change was in scope.

## 2. What WO#25 actually delivered (completed, in scope)

| File | Before | After | Change |
|---|---|---|---|
| `domains/journal/routers/journal.py` | 446 lines (all endpoints + all helpers) | 275 lines | Kept `journal_home`, `journal_date`, `save_entry`, `lock_entry`. Calendar helpers and synthesis endpoints removed (moved out). |
| `domains/journal/routers/_calendar.py` | *(new)* | 150 lines | `_build_calendar_months`, `_mood_class`, `_calculate_streak`, `_get_calendar_dates`, `_get_calendar_data` — pure helpers, no endpoints. |
| `domains/journal/routers/journal_synthesis.py` | *(new)* | 110 lines | `synthesis_history`, `latest_synthesis_json`, `synthesis_detail` — moved onto their own `APIRouter(prefix="/journal/synthesis")`. |
| `main.py` | — | — | Two-line addition: import `journal_synthesis` alongside `journal`; register `journal_synthesis.router` immediately before `journal.router` (specific-before-catch-all, matching the existing `ci_files`/`ci_readme`-before-`ci_projects` and `recipe_extract`/`recipe_discovery`/`pantry`-before-`recipes` convention). |

All three resulting files are under the 300-line ceiling (verified via
`wc -l` against a reconstructed baseline — see §4 for the caveat on why
`wc -l` was used instead of the project's actual lint script).

**Verified mechanically, not just asserted:**
- Both privacy-boundary comments (`# PRIVACY: content, gratitude,
  challenges are stored locally only...` inside `save_entry()`, and the
  `Privacy contract: ...` line in `journal.py`'s module docstring) are
  present **verbatim**, unmoved, since `save_entry()` itself was not
  relocated.
- The `domains.planning.models` local import inside `save_entry()` is
  **byte-identical** to the source — untouched, per HARD BOUNDARY.
- Route registration order inside `journal_synthesis.py` preserves the
  original ordering (`/history`, `/latest` before the parameterized
  `/{week_start_date}`) — this order matters (a literal segment route
  must be registered before a same-position parameterized route or the
  parameterized route will shadow it), and the original file already had
  it in the correct order, so this was a preservation, not a fix.
- All three files parse cleanly (`ast.parse`).

## 3. What WO#25 deliberately did *not* touch (scope boundaries held)

These are **not gaps** — they were explicit HARD BOUNDARIES or natural
consequences of "router split only," and should not cause a reviewer to
withhold sign-off:

- **`domains/journal/models.py`** — not opened, not touched. WO#25 was a
  router split; model changes were never in SCOPE.
- **The `domains.planning.models` local/deferred import** inside
  `save_entry()` — explicitly called out in WO#25's HARD BOUNDARIES as
  "a deliberate, already-reviewed cross-domain reference (see WO#6 and
  WO#10's own HARD BOUNDARIES) and unrelated to line count." Left
  completely alone. **This is the main item carried into §5 below.**
- **`routers/dashboard.py`** (the sanctioned cross-domain reader per
  GOVERNANCE.md §2.2) — not opened. Whether it imports `journal.models`
  directly or through a legacy shim was not checked, because it wasn't in
  SCOPE.
- **No schema or DB migration** — none needed, none made.

## 4. What was agreed to be deferred (explicitly out of scope for WO#25, not a defect)

Per GOVERNANCE.md §4.3's "HANDLING PRE-EXISTING BUGS" and §4.5 ("Bugs
Found During Migration Are Not Migration Work"), the following were
found but intentionally **not** fixed, and should be filed as their own
tickets rather than folded into a future migration diff:

1. **`_get_calendar_dates()` has a type-hint/implementation mismatch.**
   Annotated `-> set[datetime.date]` but its body returns a dict
   comprehension (`{date: mood_score}`). It also does not appear to be
   called anywhere in the router — likely dead code. Moved verbatim into
   `_calendar.py`, not fixed.
2. **`and_` is imported but never used** in `journal.py`, before or
   after the split. Left in place rather than silently dropped, per
   "pure relocation" principle.
3. **Live functional verification of the planning cross-domain import**
   (WO#25's own acceptance criterion: "trigger a save while a plan day
   exists, if feasible") — marked ⚠️, not ✅ or ❌. This session had no
   live FastAPI app, database, or `domains.planning` module to execute
   against — only the pasted file contents. What was confirmed is that
   the import line and its surrounding code are byte-identical to the
   source. **A real functional pass against a live environment is still
   needed before this migration is fully "done" per GOVERNANCE.md §4.6.**
4. **`scripts/check_router_line_limits.py` was not actually run** — it
   isn't available in this session (no live repo, no network). `wc -l`
   against a reconstructed baseline was used as a substitute, which
   answers the same question (line count vs. 300) but not necessarily
   with the exact same logic as the project's lint script (e.g. it may
   exclude blank lines, docstrings, or comments differently). **Should be
   re-run for real as part of CI/merge.**

None of items 1–4 block WO#25's own acceptance criteria — they're
flagged here so they don't get silently forgotten, and so a reviewer
doesn't mistake "flagged ⚠️ and explained" for "incomplete."

## 5. Follow-up work: to be done once all domain migrations are complete

This is the section this document exists to provide. It is intentionally
broader than "fix the TODO" — it's a checklist for whichever future work
order does the cross-domain cleanup pass, once `planning` (and any other
still-in-flux domain) is confirmed fully migrated.

### 5.1 Re-evaluate the deferred `domains.planning.models` import in `save_entry()`

This is the concrete "adjust the models.py, remove the reference" item.

**Current state:** Inside `save_entry()`, the import is *function-local*
and wrapped in `try/except Exception: pass` — not a normal top-level
module import:

```python
try:
    from sqlalchemy import select as _select
    from domains.planning.models import WeeklyPlanDay as _WPD, WeeklyPlan as _WP, WeeklyPlanStatus as _WPS
    ...
except Exception:
    pass  # Don't fail the journal save if this linking fails
```

**Why this matters:** GOVERNANCE.md §2.2 states the *sanctioned* pattern
for cross-domain model relationships is SQLAlchemy `relationship()` with
**string class names** (resolved via the shared mapper registry at query
time), not direct cross-domain `models.py` imports — the precedent being
`BlogIdea.code_file` / `CodeFile.blog_ideas`. The journal→planning link
does not follow that sanctioned pattern at all: it does a direct,
defensive, deferred import specifically so a not-fully-migrated or
uncertain `planning` domain can't break journal's save path at import
time or at runtime.

**Action items for the follow-up work order**, in order:

1. **Confirm `planning` is actually fully migrated** — own
   `domains/planning/models.py` (not a root-level shim), routers,
   templates, and static assets all under `domains/planning/`, per the
   GOVERNANCE.md §4.6 "Done" checklist applied to the *planning* domain
   specifically (WO#25 only ever inspected `journal`).
2. **Decide the target pattern** for the journal↔planning link:
   - **Option A — promote to a real `relationship()`.** If
     `WeeklyPlanDay.journal_entry_id` is a genuine FK column (needs
     confirming — not visible in WO#25's SCOPE), convert this into a
     proper cross-domain `relationship()` using string class names, the
     same way `blog`/`code_intel` do it. Per §2.2, **this requires
     journal and planning to be migrated together in the same work
     order** if this path is chosen (precedent: blog + code_intel in
     WO#2) — so this specific follow-up cannot be scoped as
     journal-only.
   - **Option B — keep it a direct model import, but promote it from
     function-local/defensive to a normal top-level import.** Appropriate
     if the deferred/defensive style was purely a hedge against
     `planning` being mid-migration, and that risk is now gone. This
     preserves current behavior (a soft, non-blocking link) while
     removing the historical workaround.
   - Either way, **do not silently pick one** — this changes a
     deliberately-reviewed pattern (WO#6/WO#10's own HARD BOUNDARIES
     called it out by name), so the choice itself needs its own
     reviewer sign-off, not just a diff.
3. **If Option A is chosen:** update `domains/journal/models.py` to add
   the `relationship()` side (or confirm which side it belongs on), and
   `domains/planning/models.py` symmetrically — this is the "adjust
   models.py" step. Confirm `both domains' models.py get imported
   somewhere before the first query runs" (§2.2's requirement) is
   satisfied by whatever import-registration mechanism replaced the old
   shim system (see §5.2 below) before removing the try/except.
4. **If Option B is chosen:** move the import to the top of
   `journal.py`, delete the `try/except Exception: pass` guard (or
   replace it with a narrower, intentional error-handling strategy if
   the team still wants save-doesn't-fail-if-linking-fails behavior —
   that's a product decision, not a mechanical one), and update the
   inline comment that currently justifies the defensive try/except.
5. Either way: **this is explicitly not a WO#25 deliverable.** It was
   carved out by name in WO#25's HARD BOUNDARIES precisely so this
   migration-mechanics work order wouldn't accidentally touch a
   cross-domain coupling decision. Don't retroactively treat WO#25 as
   incomplete for not doing this.

### 5.2 Reconcile GOVERNANCE.md's own internal inconsistency about `journal`/`planning` migration status

While reviewing WO#25's scope, two parts of GOVERNANCE.md appear to
disagree with each other and with the actual file layout:

- **§2.4 (Legacy Import Shims)** says shim removal is *"historical/closed"*
  — WO#20 removed every remaining domain shim from root `models.py`, and
  WO#22 removed root `models.py` itself.
- **§3.3 (Migration Debt Tracker)** still lists `journal` and `planning`
  (along with `finance`, `recipes`/`pantry`, `workout`, `media`) as
  domains *"not yet moved into the `domains/` structure."*
- **But** the files actually reviewed for WO#25 show `domains/journal/models.py`,
  `domains/journal/routers/`, `domains/journal/templates/`, and
  `domains/journal/static/` already exist, are already wired into
  `main.py` (tagged `# WO6`) and into `core/templating.py`'s
  `ChoiceLoader`. The `domains.planning.models` import inside
  `save_entry()` also already resolves to a domain-scoped path, not a
  root-level shim path — and `main.py` already imports `planning`'s
  routers from `domains.planning.routers` (tagged `# WO10`).

This is a **documentation drift**, not something WO#25 introduced or is
positioned to fix — WO#25 only had SCOPE to touch the router file. But it
should be resolved as part of (or before) whatever work order does the
follow-up in §5.1, because:

- A future agent reading only §3.3 would wrongly conclude `journal` and
  `planning` need a full ground-up domain-folder migration (models +
  routers + templates + static), when the routers/templates/static
  clearly already happened (WO#6, WO#10) and the models likely did too.
- Getting this wrong would cause a future work order to duplicate
  already-done work, or — worse — to treat the already-reviewed
  `domains.planning.models` import as if it were an unmigrated,
  temporary shim reference rather than the deliberate design choice it
  actually is.

**Action item:** once `planning`'s full migration status is confirmed
one way or the other (§5.1 step 1), update GOVERNANCE.md §3.3 to remove
`journal` and, if applicable, `planning` from the debt tracker, or
correct the record if there's a `models.py`-specific gap that the
router/template migrations didn't actually close.

### 5.3 Confirm no root-level shim remnants still reference journal or planning

- Per §2.4, shims were supposed to be fully retired once
  `routers/dashboard.py` (the one sanctioned cross-domain reader) imports
  directly from each domain's `models.py`.
- WO#25 did not open `routers/dashboard.py` — it was outside SCOPE.
- **Action item:** confirm `dashboard.py` imports `JournalEntry` /
  `WeeklySynthesis` (and any `planning` models it reads) directly from
  `domains.journal.models` / `domains.planning.models`, with no
  lingering root-level shim in between. If a shim still exists, retire it
  per the §2.4 process (own small cleanup task, not bundled into a
  migration diff).

### 5.4 Re-run the domain-level "Done" checklist (GOVERNANCE.md §4.6) for `journal` as a whole

WO#25 re-verified only the **router-split** slice of "done" (line counts,
endpoint shapes, privacy comments). It did not re-confirm the full
domain-migration checklist for `journal`:

- [ ] Models, routers, templates, and static assets all live under
      `domains/journal/` — *appears true from the files reviewed, but
      was never the object of WO#25's verification; re-confirm formally.*
- [x] `main.py` and `core/templating.py` reference the new paths — true
      for templates already; `main.py` was updated by WO#25 itself for
      the new `journal_synthesis` sub-router.
- [ ] Legacy shim exists (or has been correctly retired) for any external
      consumer (normally just `dashboard.py`) — **not checked**, see §5.3.
- [x] WO#25's own acceptance criteria passed (✅/⚠️ as tracked in §2–4
      above).
- [ ] No unrelated behavior changed, confirmed via `Base.metadata`
      identity check and functional re-verification of every affected
      endpoint — **partially done** (structural/static verification only;
      live functional re-verification is the §4 item 3 gap above).

### 5.5 Re-run the real lint script and a live functional pass

Carried over from §4 items 3–4 — repeated here because this is the
"final gate" section a reviewer should check before considering the
whole thing closed, not just WO#25 in isolation:

- Run the actual `scripts/check_router_line_limits.py` against the
  merged files in CI.
- Exercise `save_entry()` against a live DB with an existing,
  `CONFIRMED`/`ACTIVE` `WeeklyPlan` and a matching `WeeklyPlanDay` row for
  today's date, to confirm the planning linkage still fires and still
  sets `journal_entry_id` — this was never executable in the WO#25
  authoring session.

---

## 6. Reviewer sign-off guidance

Per GOVERNANCE.md §4.4's report review order:

1. **Hard boundaries respected?** Yes — `domains/journal/models.py`, the
   `domains.planning.models` import, and every endpoint's path/
   method/shape are all untouched or byte-identical. Only
   `domains/journal/routers/*.py` and `main.py` were edited/created, per
   SCOPE.
2. **Are the ❌/⚠️ items genuinely outside this WO's control?** Yes — the
   two ⚠️ items in §4 (functional planning-link test, real lint script)
   are both blocked by this being a document-only authoring session with
   no live app, DB, or repo checkout — not by incomplete work.
3. **Does anything here need its own ticket?** Yes — §4 items 1–2
   (pre-existing bugs) and all of §5 (deferred cross-domain cleanup,
   documentation drift) should be filed as separate, standalone tickets,
   not bundled into whatever work order eventually does them.
4. **Do the acceptance criteria that matter functionally pass?** Yes, to
   the extent staticly verifiable in this session — see §2's
   "Verified mechanically" list.

**Recommendation: approve WO#25 as complete on its own terms.** None of
§5's items were ever part of its deliverable, and treating them as
blocking would conflate a 300-line router split with an unrelated
cross-domain architecture decision that GOVERNANCE.md itself says
requires its own, jointly-scoped work order (§2.2).
