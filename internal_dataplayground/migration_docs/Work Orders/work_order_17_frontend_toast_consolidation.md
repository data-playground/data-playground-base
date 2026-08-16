# Work Order #17 — Frontend Consolidation: Toast Notifications + Dead Sidebar JS

*Addresses the toast-notification duplication tracked in GOVERNANCE.md
§3.2 (first item on that list) plus a likely-dead file
(`partials/sidebar_js.html`) that fully duplicates functions already in
`base.js`. This is JS/template cleanup only — no inline-CSS-style cleanup
is in scope here, since GOVERNANCE.md §3.2 explicitly does NOT call for
retroactive inline-style fixes across existing templates (only newly
written/edited code is held to that standard).*

---

## ROLE
You are a senior refactoring engineer removing duplicated frontend code.
Your job is to make five pages (plus one dead file) use the shared
`#toast` element and `showToast()` function from `base.js`/`base.html`
instead of their own local copies — not to redesign the toast system, add
new toast features, or touch anything about how these pages otherwise
work.

## HARD BOUNDARIES
- Only read/edit files explicitly listed in SCOPE below.
- **The shared `showToast()` in `base.js` uses a 2600ms display duration.**
  The five local duplicates being removed use slightly different
  durations (2400ms in `recipes.html`/`pantry.html`, 2800ms in
  `blog.html`, 2200ms in `jobs.html`, 2400ms in `weekly_plan_view.html`).
  Consolidating onto the shared function means every page's toast will now
  display for 2600ms instead of its previous page-specific duration. **This
  is a real, disclosed behavior change** (timing only, nothing functional)
  — call it out explicitly in your report rather than treating it as an
  invisible side effect of the cleanup. If you believe this timing
  difference matters enough to preserve per-page, stop and ask rather than
  either silently standardizing it or silently preserving five different
  durations by leaving the duplicates in place — that decision belongs to
  whoever reviews this work order, not to you.
- Do not change `base.js`'s `showToast()` implementation itself except if
  Step 1 requires adding something it's currently missing (see Step 1) —
  and if so, that addition must not change existing behavior for any
  current caller of the shared function.
- Every page in SCOPE already extends `base.html`, which unconditionally
  renders `<div id="toast"></div>` and includes `base.js`. Confirm this is
  true for each page before removing its local toast `<div>` — if any page
  in SCOPE turns out not to extend `base.html` the way assumed, stop and
  report rather than assuming the shared element is available.

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
Execute steps in order. Verify incrementally, not only at the end.

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

**Templates to edit (remove local toast duplication):**
- `templates/recipes.html` (local `#recipe-toast` div +
  `showRecipeToast()` function)
- `templates/pantry.html` (local `#pantry-toast` div +
  `showPantryToast()` function)
- `templates/blog.html` (local `showToast()` function redefinition — uses
  the shared `#toast` element already, just redefines the function
  unnecessarily)
- `templates/jobs.html` (same pattern as `blog.html`)
- `templates/weekly_plan_view.html` (same pattern as `blog.html`)

*(Note: if WO#3, WO#5/#7, or WO#10 have already run by the time this work
order executes, `jobs.html`, `recipes.html`/`pantry.html`, and
`weekly_plan_view.html` may be under `domains/<name>/templates/` instead
of the root `templates/` directory — locate each wherever it currently is;
this work order does not depend on any prior domain migration having run
first.)*

**File to investigate and likely delete:**
- `templates/partials/sidebar_js.html`

**Reference only, do not edit:**
- `static/js/base.js` (the canonical `showToast()` implementation)
- `templates/base.html` (confirms the shared `#toast` div and `base.js`
  inclusion)

---

## STEPS

1. **Confirm `base.js`'s `showToast(msg, isError = false)` covers every
   usage pattern the five duplicates need.** Check each duplicate's call
   sites: do any of them call their local toast function with arguments
   or behavior the shared one doesn't support (e.g. a third parameter, a
   different error-state class name)? Based on the code as currently
   written, all five appear to be simple `(message)` or `(message,
   isError)` calls compatible with the shared signature — confirm this
   rather than assuming, and if you find an incompatible usage, stop and
   report rather than either breaking that call site or silently adding
   unrequested capability to `base.js`.

2. **`recipes.html`:** delete the `<div id="recipe-toast"></div>` element
   and the local `showRecipeToast()` function definition. Replace every
   call site (`showRecipeToast(...)`) with `showToast(...)`, matching
   argument order.

3. **`pantry.html`:** delete the `<div id="pantry-toast"></div>` element
   and the local `showPantryToast()` function definition. Replace every
   call site (`showPantryToast(...)`) with `showToast(...)`.

4. **`blog.html`, `jobs.html`, `weekly_plan_view.html`:** delete each
   page's local `showToast()` function redefinition (including its
   `let toastTimer` / `_t` local timer variable, if the page declares
   one). Do NOT delete any `#toast` `<div>` in these three files unless
   you confirm one exists locally in addition to the inherited one from
   `base.html` (based on the current code, these three don't declare a
   local `<div id="toast">` — they only redefine the function — but
   verify this per-file rather than assuming all three match exactly).
   No call sites need updating in these three files, since they already
   call `showToast(...)` — only the redundant local definition is removed,
   so calls now resolve to `base.js`'s version instead.

5. **Investigate `templates/partials/sidebar_js.html`.** Search the whole
   codebase for any `{% include "partials/sidebar_js.html" %}` (or
   equivalent Jinja2 include/import) reference. Its own docstring claims
   it's meant to be included "at the bottom of every standalone page's
   `<script>` block" for pages that don't extend `base.html` — but every
   template reviewed so far extends `base.html`, which already includes
   `base.js` (containing the same `setTheme`, `toggleSidebar`,
   `applyCollapsed`, `openMobileSidebar`, `closeMobileSidebar` functions
   this file redefines). If the search confirms **zero** includes
   anywhere in the codebase, delete the file. If you find even one
   include site, do NOT delete the file — instead, remove that one
   include and confirm the page that referenced it still works correctly
   via its inherited `base.js` (since it necessarily extends `base.html`
   for the toast/sidebar infrastructure to exist at all), and report which
   page needed that change.

---

## ACCEPTANCE CRITERIA

- [ ] `GET /recipes` and `GET /pantry` — trigger an action that shows a
  toast (e.g. favoriting a recipe, adding a pantry ingredient) and confirm
  it displays correctly using the shared `#toast` element, with the
  2600ms shared duration (not the previous 2400ms)
- [ ] `GET /blog`, `GET /jobs`, and the weekly plan view page — trigger a
  toast-producing action on each (e.g. archiving a blog idea, logging an
  ATS status, marking a meal eaten) and confirm each still displays
  correctly via the shared function
- [ ] No page in SCOPE has a duplicate `<div id="toast">`-equivalent
  element remaining after this change — confirm via `grep` for
  `recipe-toast` and `pantry-toast` returning zero hits anywhere in the
  codebase
- [ ] No page in SCOPE has a local `function showToast(` definition
  remaining — confirm via `grep -n "function showToast"` across all
  template files returning exactly one hit (`base.js`)
- [ ] `templates/partials/sidebar_js.html` is either deleted (if zero
  includes found) or has exactly the one confirmed include site updated
  (if any were found) — report which outcome occurred and the evidence
  for it
- [ ] Every other feature on the five edited pages (unrelated to toasts)
  renders and functions identically — spot-check at least one non-toast
  interaction per page to confirm the edits didn't have unintended
  side effects
- [ ] The 2200–2800ms → 2600ms timing harmonization is explicitly called
  out in your report per HARD BOUNDARIES, not just silently reflected in
  the diff

---

## For the next work order (not part of this one)

No further frontend work is currently queued as its own dedicated work
order — per GOVERNANCE.md §3.2, inline-style/CSS-primitive cleanup is
enforced going forward on new and edited code, not retroactively across
the whole template set, so there's no "fix all inline styles" work order
to write. If a future domain migration or feature touches a template
heavily enough that its accumulated inline-style duplication becomes worth
addressing in the same pass, handle it there as an explicitly-called-out
exception (same treatment WO#10 gave `weekly_plan.py`'s 300-line split),
not as a separate standalone project.

With this work order, every item explicitly tracked in GOVERNANCE.md §3.2
("Known Duplication Still Being Worked Down") except the still-open AI
service layer retrofitting question (deliberately left alone per WO#16)
is now resolved. The remaining open threads across this whole engagement
are: running WO#2–10 and WO#12–17 (only WO#1 and this drafting work is
"real" so far — execution is still pending for everything else), the
deferred DAG relocation phase, and the four backlog ideas in
GOVERNANCE.md §5.
