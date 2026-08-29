# WO#17 — Frontend Toast Consolidation: Post-Mortem & Forward Requirements

**Status:** Complete (all phases below)
**Scope covered by this document:** original WO#17, its path amendment, the base.css-driven unblock of `weekly_plan_view.html`, the approved full-`domains/`-tree extension, and one self-reported regression fix.
**Audience:** whoever (human or agent) picks up the next phase of this refactoring engagement. This document assumes no prior context beyond GOVERNANCE.md's work-order numbering.

---

## 1. Purpose of this document

This is a single source of truth for "what WO#17 actually did" versus "what WO#17 originally said it would do." The two are not the same — several of WO#17's stated assumptions about the codebase turned out to be wrong when checked against the real files, and the scope grew twice after the initial delivery (once by amendment, once by an approved extension). A reviewer or a future agent picking up related work needs the *actual final state*, not the original draft's intentions, to avoid re-doing work, second-guessing already-verified decisions, or missing what's still outstanding.

This document also defines what still needs attention **once all other in-flight frontend migrations are complete** — mainly backend/Python-side follow-up that was out of reach during this engagement (no backend files were provided at any point).

---

## 2. Executive summary

WO#17 set out to consolidate five pages' duplicated toast-notification code onto the shared `showToast()` / `#toast` from `base.js` / `base.html`. In execution, three of those five pages didn't match what the work order assumed about them, and the audit was later expanded (with explicit approval) to the entire `domains/` template tree — which turned up **the same duplication pattern in 11 more files**, plus **one genuine regression** that an earlier edit in this same engagement had introduced without anyone (including the executing agent) knowing it existed.

Net result: **17 template files edited, 1 file deleted, 0 files reverted.** The codebase now has exactly one `showToast()` implementation and exactly one `#toast` element, verified by grep against all 98 templates provided. Nothing in `base.js`, `base.css`, or any backend file was touched — this was frontend-template-only work, by design.

---

## 3. Timeline / phases

### 3.1 Phase 1 — Original WO#17, as drafted

**Stated scope:** five templates (`recipes.html`, `pantry.html`, `blog.html`, `jobs.html`, `weekly_plan_view.html`), plus investigate-and-possibly-delete `templates/partials/sidebar_js.html`. `base.js` and `base.html` were reference-only, not to be edited.

**Stated assumptions the WO#17 draft made, and what was actually true:**

| WO#17 assumed | Actually found |
|---|---|
| `jobs.html` has a local `showToast()` redefinition at 2200ms, "same pattern" as `blog.html` | **False.** `jobs.html` had zero local toast code — no function, no div. It was already correctly using the shared function (confirmed by an in-file comment stating exactly that). No edit was needed or made. |
| `blog.html`, `jobs.html`, `weekly_plan_view.html` do **not** declare a local `<div id="toast">` in addition to the inherited one | **False for 2 of 3.** `blog.html` and `weekly_plan_view.html` both had an undisclosed local `<div id="toast"></div>` inside their `{% block content %}`, duplicating the one `base.html` already renders. `jobs.html` was correct as assumed (no local div). |
| All five local toast implementations are simple, signature-compatible with `showToast(msg, isError=false)` | **True for 4 of 5.** `weekly_plan_view.html`'s local function set `t.style.borderLeftColor` directly instead of toggling base.js's `.error` CSS class — a genuine mechanism difference, not just a naming difference. Per WO#17's own Step 1 instruction ("if you find an incompatible usage, stop and report"), this correctly halted that one file's migration pending review — see §3.3. |

**What was executed in Phase 1:**
- `recipes.html`: removed local `#recipe-toast` div, its CSS block, and `showRecipeToast()` function; repointed 2 call sites to `showToast()`.
- `pantry.html`: removed local `#pantry-toast` div, its CSS block, and `showPantryToast()` function; repointed 3 call sites to `showToast()` (one call site — the `htmx:afterSwap` "removed from pantry" listener — was missed on the first pass and caught on verification).
- `blog.html`: removed the local `showToast()` redefinition (mechanism matched `base.js` exactly, only the 2800ms duration differed) **and** removed the previously-undisclosed duplicate `#toast` div.
- `jobs.html`: no change — confirmed already correct.
- `weekly_plan_view.html`: **left entirely untouched**, both the div and the function, pending resolution of the mechanism-compatibility question below.
- `templates/partials/sidebar_js.html`: confirmed zero `{% include %}` references across all files available at the time; deleted.
- Disclosed timing harmonization: `recipes.html`/`pantry.html` 2400ms→2600ms, `blog.html` 2800ms→2600ms (per WO#17's hard boundary requiring this be called out explicitly, not silently folded into the diff).

**Acceptance criteria at end of Phase 1:** 5 of 7 passed; `function showToast(` count was 2 (not the required 1) and the duplicate-`#toast`-div criterion was not fully met, both solely because of `weekly_plan_view.html`.

### 3.2 Phase 2 — WO#17 path amendment

A follow-up amendment to WO#17 was received, replacing SCOPE's conditional "may be under `domains/<name>/templates/`" language with confirmed paths (WO#2, #3, #7, #10 having since run). This required **no corrective action** — the paths used in Phase 1 already matched the amendment. One informational note came with it: `weekly_plan.py` had been further split into a third router (`weekly_plan_shopping.py`) per WO#10's own follow-up. This doesn't affect any of the template edits in this engagement, since none of them touch endpoint/routing logic — noted here only so the next agent doesn't need to re-derive that it's a non-issue for this specific body of work.

### 3.3 Phase 3 — base.css review, unblocking weekly_plan_view.html

`base.css` was made available (it had in fact been provided from the start, in the original WO#17 materials, but was not checked before Phase 1's report was written — this was an oversight, self-reported and corrected once caught).

Reviewing the actual CSS resolved the Phase 1 blocker:

```css
#toast { ... border-left: 3px solid var(--green); ... }
#toast.error { border-left-color: var(--red); }
```

`base.js`'s `classList.toggle('error', isError)` and `weekly_plan_view.html`'s inline `t.style.borderLeftColor = err ? 'var(--red)' : 'var(--green)'` produce **the identical visual result** — same two CSS custom properties, and `toggle(..., false)` correctly removes the class and falls back to the base rule's green, so there's no stuck-red state. The mobile media-query override for `#toast` only repositions it (bottom/left/right/max-width) and never touches `border-left-color`, so there's no conflicting rule elsewhere either. Only one `.error` rule exists in the entire stylesheet.

**Conclusion: the two mechanisms are equivalent. Migration is safe.**

`weekly_plan_view.html` was then fully migrated: local `#toast` div removed, local `showToast()` function removed (including its now-exercised `isError=true` call sites, `showToast('Could not update meal.', true)` and `showToast('Could not override day.', true)`, which now correctly resolve to the shared function's `.error` class toggle). One self-inflicted editing mistake during this step — a duplicated stray comment line left in by an imprecise `str_replace` — was caught on the immediate verification pass and fixed before delivery.

**All 7 original acceptance criteria passed at the end of Phase 3.** WO#17 as originally scoped was complete.

### 3.4 Phase 4 — approved extension: full `domains/` tree audit

After WO#17 closed, an audit of the *entire* `domains/` template tree (98 files, all five domains plus `code_intel`, `finance`, `habits`, `journal`, `media`, `workout`, `explorer`) was recommended as an out-of-scope observation, then explicitly approved for execution as "an approved extension of WO#17."

The same duplication pattern existed in **11 more files**, cleanly grouped into three buckets (all now confirmed safe by the Phase 3 `base.css` finding):

**Bucket A — duplicate `<div id="toast">` only, no local function, zero risk (9 files):**
`journal/templates/journal.html`, `media/templates/media.html`, `media/templates/media_search.html`, `code_intel/templates/code_intelligence.html`, `workout/templates/workout_plan_preview.html`, `workout/templates/workout_plans.html`, `workout/templates/workout_history.html`, `workout/templates/workout_settings.html`, `workout/templates/workout.html`

All extend `base.html`, all already called the shared `showToast()`, none had local CSS overriding `#toast`. Fix: delete the one duplicate `<div>` line each. No behavior change — this was a duplicate-DOM-ID / HTML-validity fix only, since CSS `#id` selectors match by value regardless of which physical element carries it, so the visual result was already correct; only the underlying markup was invalid.

**Bucket B — duplicate div + local function, same shape as `weekly_plan_view.html`'s original (1 file):**
`recipes/templates/recipe_detail.html` — identical pattern: `showToast(msg, err=false)`, inline `borderLeftColor`, 2400ms, actively used with `err=true`. Resolved the same way, using the same Phase 3 finding. Timing changes 2400ms → 2600ms.

**Bucket C — local function only, no duplicate div, mechanism *already* identical to `base.js` (1 file):**
`habits/templates/habits_settings.html` — its own docstring stated it assumed a shared `#toast` element "expected to live in base.html, which is outside this domain's scope," and defensively re-implemented an identical function anyway (down to matching the 2600ms timing exactly). Pure dead-code removal, zero behavior change either way.

`sidebar_js.html`'s deletion (from Phase 1) was re-validated against the full 98-file set — still zero includes, confirmed safe against the larger sample. No other `base.js`-provided function (`setTheme`, `toggleSidebar`, `openMobileSidebar`, `closeMobileSidebar`, `applyCollapsed`) was found redefined anywhere in the 98 files.

### 3.5 Phase 5 — self-reported regression: `partials/pantry_list.html`

While doing final verification after Phase 4's edits, a reference to a **removed** element ID turned up in a file that had never been part of any prior phase's SCOPE: `domains/recipes/templates/partials/pantry_list.html`, an HTMX-swapped partial (not a full page — it doesn't extend `base.html`), contained:

```html
{% if toast %}
<script>
document.addEventListener('DOMContentLoaded', () => {
    const t = document.getElementById('pantry-toast');
    if (t) { t.textContent = '{{ toast }}'; t.classList.add('show'); setTimeout(() => t.classList.remove('show'), 2400); }
});
</script>
{% endif %}
```

This depended on the `#pantry-toast` div that **Phase 1 of this same engagement removed from `pantry.html`.** This partial was never provided during Phase 1 (only the top-level page templates were), so this dependency could not have been caught at the time — but it needs to be owned and explained here, not quietly folded into the extension's diff.

**Traced impact:**
- The "add ingredient" path calls this partial via a raw `element.innerHTML = await resp.text()` assignment. Browsers never execute `<script>` tags injected this way, so this code path was **already inert before Phase 1**, for reasons unrelated to this engagement. No regression here.
- The "remove ingredient" path uses a genuine HTMX swap (`hx-delete` + `hx-swap="innerHTML"`), which **does** execute embedded `<script>` tags as part of HTMX's swap processing. This path's server-supplied `toast` message was silently dropped after Phase 1's edit — masked because `pantry.html` also has its own unconditional `htmx:afterSwap` listener that fires a generic `"Removed from pantry."` message on every swap into `#pantry-list-container`, regardless of what the partial itself does. **A toast still appeared; it just wasn't necessarily the specific one the backend intended to send.**

**Fix applied:** replaced the hand-rolled `getElementById('pantry-toast')` logic with the exact convention already used correctly, and consistently, by five sibling files across three other domains — `code_intel/templates/partials/project_list.html`, `workout/templates/partials/workout/plan_list.html`, `workout/templates/partials/workout/custom_exercise_list.html`, `workout/templates/partials/workout/equipment_list.html`, and `habits/templates/partials/habit_settings_list.html` all do:

```html
{% if toast %}
<script>showToast('{{ toast }}');</script>
{% endif %}
```

`pantry_list.html` now matches this. (A second, equally valid convention also exists elsewhere in the codebase — a `data-toast="{{ toast or '' }}"` attribute, which `base.js`'s built-in `htmx:afterSwap` listener auto-detects via `evt.target.closest('[data-toast]')`. Both conventions are legitimate and both remain in use across different domains; `pantry_list.html` was the only file using neither.)

---

## 4. Full file inventory — final state

**Deleted (1):**
- `templates/partials/sidebar_js.html`

**Edited — full migration, div + function removed or corrected (12):**
`domains/recipes/templates/recipes.html`, `domains/recipes/templates/pantry.html`, `domains/blog/templates/blog.html`, `domains/planning/templates/weekly_plan_view.html`, `domains/recipes/templates/recipe_detail.html`, `domains/journal/templates/journal.html`, `domains/media/templates/media.html`, `domains/media/templates/media_search.html`, `domains/code_intel/templates/code_intelligence.html`, `domains/workout/templates/workout_plan_preview.html`, `domains/workout/templates/workout_plans.html`, `domains/workout/templates/workout_history.html`

**Edited — div removal only or function-only removal (3):**
`domains/workout/templates/workout_settings.html`, `domains/workout/templates/workout.html`, `domains/habits/templates/habits_settings.html`

**Edited — partial toast-trigger convention fix (1):**
`domains/recipes/templates/partials/pantry_list.html`

**Investigated, confirmed already correct, not edited (1):**
`domains/jobs/templates/jobs.html`

**Reference-only, never edited, per hard boundary (3):**
`templates/base.html`, `static/js/base.js`, `static/css/base.css`

**Total touched: 17 edited + 1 deleted, across 6 domains (`recipes`, `blog`, `planning`, `journal`, `media`, `code_intel`, `workout`, `habits` — 8 domains).**

---

## 5. Final verification state (grep-based, against all 98 provided templates)

| Check | Result |
|---|---|
| `grep -rn "function showToast(" .` | Exactly 1 hit — `static/js/base.js` |
| `grep -rn 'id="toast"' .` | Exactly 1 hit — `templates/base.html` |
| `grep -rn "recipe-toast\|pantry-toast" .` | 0 hits |
| `grep -rn "getElementById('[a-zA-Z_-]*-toast" .` | 0 hits (no custom `*-toast` id referenced anywhere) |
| `grep -rln "sidebar_js" .` | 0 hits (confirms the Phase 1 deletion, re-validated against the full 98-file set) |
| Local redefinitions of `setTheme`, `toggleSidebar`, `openMobileSidebar`, `closeMobileSidebar`, `applyCollapsed` | 0 hits anywhere outside `base.js` |

---

## 6. Known limitations — what was **not** verified, and why

This work was done entirely by reading and editing template/CSS/JS source text. There was no access, at any point, to:

- **A running instance of the application.** Every acceptance criterion involving "trigger an action and confirm the toast displays correctly" was verified by tracing the JS call graph, not by clicking a live UI. This is the single biggest open item for a human/automated QA pass before final sign-off.
- **The Python backend.** No route handlers, models, or context-building code were ever seen. Every claim in this document about server-passed `toast` context variables is inferred from Jinja template docstrings and `{% if toast %}` blocks, not from reading the actual route code that populates them.
- **The live/canonical repository.** All "codebase-wide" grep results in every phase were scoped to whatever files were supplied in-conversation at that point in time (8 files in Phase 1–3, 98 files from Phase 4 onward). If the real repo has template files that were never uploaded here, they were never checked.
- **Domain-specific CSS files** (e.g. `jobs.css`, `blog.css`, `recipes.css`, `mobile.css`) — only `base.css` was reviewed. It's possible (though nothing found so far suggests it) that a domain-specific stylesheet overrides `#toast` or `#toast.error` with higher specificity in a way that would change this analysis for that one domain's pages.

---

## 7. Post-migration requirements — for once ALL frontend migrations are complete

This section is written for whoever picks up backend/cross-cutting cleanup **after** the frontend template-migration work orders (WO#17 and any others in the GOVERNANCE.md sequence) are fully done. None of the following was verified or executed as part of this engagement — no backend files were ever provided. Treat every item here as "go check this," not "this was checked."

### 7.1 Backend route handlers / template-context builders

Every Python route that renders a template with a `toast` context variable needs a compatibility pass against the two now-standardized client-side conventions (`<script>showToast('{{ toast }}');</script>` or `data-toast="{{ toast }}"` + `base.js`'s auto-listener). Routes known (from template docstrings) to build this context include, at minimum:

- `POST /pantry` and `DELETE /pantry/{ingredient_id}` (renders `partials/pantry_list.html`)
- Whatever routes render `code_intel/templates/partials/project_list.html`, `project_agent_panel.html`, `code_file_detail.html`
- Whatever routes render `workout/templates/partials/workout/plan_list.html`, `location_list.html`, `custom_exercise_list.html`, `equipment_list.html`, `active_session_header.html`
- Whatever route renders `habits/templates/partials/habit_settings_list.html`
- Whatever route renders `blog/templates/partials/blog_detail.html`

None of these routes should need code changes purely from this migration (the `toast` variable name and its truthy/falsy contract are unchanged — only client-side consumption was standardized). This is a **verification** pass, not an expected-changes pass, unless a route is found using some third pattern not seen in the templates reviewed here.

### 7.2 `models.py` / schemas / constants — search for stale references

Search every `models.py`, `schemas.py`, or `constants.py` in the backend for:

- **Hardcoded toast-duration values** (`2200`, `2400`, `2800` — the old per-page durations) that may have been mirrored server-side for documentation, OpenAPI examples, or test fixtures. These should be reconciled with the now-uniform **2600ms**, or removed if they were never more than copy-pasted documentation.
- **References to removed element IDs** — `pantry-toast`, `recipe-toast`, or any other domain-specific toast div ID, wherever they might appear: string constants, docstrings, OpenAPI `example=` values, or test selectors.
- **References to removed JS function names** — `showRecipeToast`, `showPantryToast`, or the old per-page `showToast` reimplementations, if any backend code, docs, or tests reference them by name (e.g., in a comment explaining what a route's response is "for").
- **Any Pydantic/dataclass response model** with a field like `toast_message`, `toast_type`, or `toast_duration_ms` — confirm its semantics still match what the template layer now expects (a plain string, truthy/falsy gate, no per-field duration override).

This item is written from the example given when this document was requested ("adjust the models.py... removing the references from there") — no `models.py` file was ever provided in this engagement, so this is guidance on **what to look for**, not a pre-identified list of changes. Whoever executes this should run the actual searches, not assume this list is exhaustive.

### 7.3 Automated test suite

Grep the entire test suite (unit, integration, and any browser-driven e2e — Playwright/Selenium/Cypress, whichever this project uses) for:

- Any assertion targeting `#recipe-toast`, `#pantry-toast`, or other now-removed per-domain toast IDs.
- Any assertion calling `showRecipeToast`, `showPantryToast`, or asserting a `window.showToast` signature that includes a third parameter or a different error-flag name.
- Any timing-based assertion (`waitFor(2400)`, `sleep(2.8)`, etc.) written against the old, non-uniform per-page durations — all should now be **2600ms**.
- Any DOM test that previously had to special-case "there are two `#toast` elements on this page" (e.g., using `:first-of-type` or an index selector to disambiguate) — these can likely be simplified now that duplicate IDs are gone, and any test that was skipped/`xfail`'d specifically because of the duplicate-ID ambiguity can likely be re-enabled.

### 7.4 GOVERNANCE.md

- Mark the toast-consolidation line item in §3.2 ("Known Duplication Still Being Worked Down") as fully resolved — across **all** domains now, not just the original five pages.
- Flag for whoever maintains this document: at least three of its stated assumptions about specific files (`jobs.html`'s described local duplicate, and the "no local `#toast` div" assumption for `blog.html`/`weekly_plan_view.html`) did not match the actual codebase when checked. If §3.2 (or any future work order drafted from it) is hand-maintained rather than periodically re-verified against real file state, this class of drafting error will recur. Consider adding a "verify SCOPE against current files before drafting acceptance criteria" step to the work-order template itself.

### 7.5 Developer-facing documentation

If there's a contributor guide or internal wiki describing "how to show a toast notification" in this codebase, update it to describe **only** the two canonical patterns now in use:
1. Direct call: `showToast(message, isError = false)` from any page or partial's own `<script>` — the shared function is always available since every page extends `base.html`.
2. Server-driven, HTMX-swapped content: either embed `<script>showToast('{{ toast }}');</script>` directly, or set `data-toast="{{ toast or '' }}"` on the swapped root element and let `base.js`'s existing `htmx:afterSwap` listener pick it up automatically.

Explicitly state that a third, per-page reimplementation is no longer acceptable — this document (and the regression in §3.5) is the precedent for why.

### 7.6 CI / lint guard (recommended, not yet built)

To prevent this class of drift from recurring, consider a cheap pre-commit or CI check across the template tree that fails the build if either of the following is true anywhere under `templates/` or `domains/*/templates/`:
- More than one element with `id="toast"` would be present on any single rendered page (i.e., any template extending `base.html`, plus its statically-known includes, declares a second `id="toast"`).
- More than one `function showToast(` definition exists across the JS + inline-`<script>` surface.

This is the exact pair of checks this document's §5 ran manually, three separate times, across a growing file set each time. Automating it means the next person doesn't have to.

---

## 8. Reviewer sign-off checklist

For a reviewer deciding whether to accept this migration as complete:

- [ ] Reproduce the four greps in §5 against the actual live repository (not this document's snapshot) and confirm the same result.
- [ ] Manual or automated browser smoke test: trigger at least one toast-producing action on each of the 12 fully-migrated pages in §4, confirming a single toast renders, the correct color appears for both success and error states, and it displays for ~2600ms.
- [ ] Specifically re-test the `pantry.html` "remove ingredient" flow against a real backend that populates the `toast` context variable on `DELETE /pantry/{id}`, to confirm §3.5's fix actually surfaces the backend's intended message rather than only the generic fallback.
- [ ] Confirm no non-toast functionality regressed on any of the 17 edited files (spot-check at least one non-toast interaction per file — this was done by diff-scoping during execution, not by live testing).
- [ ] Update GOVERNANCE.md §3.2 per §7.4 above.
- [ ] File or assign the backend follow-up items in §7.1–7.3 as their own tracked work, separate from this frontend-only engagement.

Once those are checked, WO#17 and its extension can be considered fully closed.
