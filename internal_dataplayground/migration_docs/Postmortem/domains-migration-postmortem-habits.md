# Habits Domain Migration — Post-Mortem & Forward Requirements

**Status:** Complete and confirmed working in production.
**Scope of this document:** (1) a post-mortem of the habits-domain pilot migration,
covering what happened, what broke, and why; (2) a checklist/spec for whoever
(human or agent) migrates each subsequent domain, so the same mistakes aren't
repeated; (3) a **final cleanup checklist** to execute only once every domain
has been migrated — several cleanup steps are unsafe to do per-domain and
must wait until the whole series is done.

This document assumes no prior conversation context. If you're an agent
picking up "migrate the `<X>` domain," read this whole document first —
specifically Section 4 (lessons learned) and Section 5 (per-domain
checklist) before touching any files.

---

## 1. What This Migration Was

**Goal:** prove a `domains/<name>/` folder-per-domain pattern on the
lowest-risk module (`habits`) before touching anything else, as a pilot for
restructuring the rest of a monolithic FastAPI app (`internal_dataplayground/`)
that currently keeps all models in one `models.py`, all routers in one
`routers/` folder, all templates in one `templates/` folder, etc.

**Target structure achieved:**
```
domains/habits/
    __init__.py
    models.py
    routers/
        __init__.py
        habits.py
    templates/
        habits.html
        habits_settings.html
        partials/
            habit_card.html
            habit_progress.html
            habit_settings_list.html
            habit_settings_row.html
    static/
        css/habits.css
```
Plus shared infrastructure created for this pilot and reused by every future
domain: `core/base_model.py` (the shared SQLAlchemy `Base`) and
`core/templating.py` (a `Jinja2Templates` instance backed by a
`jinja2.ChoiceLoader`, so templates resolve by bare filename regardless of
which domain folder they physically live in).

**Backward-compatibility mechanism:** the old `models.py` keeps a re-export
shim (`from domains.habits.models import Habit, HabitLog, HabitSettings, ...`)
so any file still doing `from models import Habit` keeps working unchanged.
Same idea for routing: `main.py` imports `from domains.habits.routers import
habits` instead of `from routers import habits`, but nothing outside
`main.py` needed to change.

---

## 2. Timeline

1. **Initial migration** (models split, router moved, templates moved, CSS
   moved, `main.py`/`core/` wiring added). All done as a pure location
   refactor — no intended behavior or schema changes.
2. **Acceptance testing surfaced a pre-existing bug**, unrelated to the
   migration: `POST/DELETE /habits/log` 500'd with `'view' is undefined`.
   Root cause: the router spread a `view` dict into the template context
   (`**view`) instead of nesting it (`"view": view`), but the partial
   template expected a nested `view` object. This bug existed in the
   original code before the migration touched it; the migration just
   happened to be the moment someone actually exercised that code path
   during structured acceptance testing.
3. **First fix ticket**: nest `view` correctly. This alone was *insufficient*
   — the same partial also references `habit.id`/`habit.color`/`habit.icon`/
   `habit.name` directly (not via `view.habit`), so the fix had to pass
   **both** `view` and `habit` into the context.
4. **Second, deeper bug found while verifying the first fix**: a duplicate
   same-day log attempt hit `db.rollback()` (via a pre-existing
   `except IntegrityError` branch), which expires all ORM objects in the
   session — including `habit` — and the code then accessed `habit.id`
   without an explicit re-fetch, triggering
   `greenlet_spawn has not been called` under async SQLAlchemy. Confirmed
   this wasn't limited to the duplicate-log edge case: under
   `expire_on_commit=True` (SQLAlchemy's own default, unknown whether the
   real `database.py` overrides it), the *plain* happy-path log/unlog would
   also crash. Fixed by re-fetching `habit` via `await db.get(...)`
   immediately before it's used in both `log_habit()` and `unlog_habit()`,
   defensively, regardless of session config.
5. **Documentation pass**: added docstrings to every previously-undocumented
   Python class/function in the touched files, JSDoc comments to every JS
   function in `habits_settings.html`, corrected a stale/actively-wrong
   template doc comment (see Section 4.3), declared a `UniqueConstraint`
   in the ORM model that had previously existed only as a comment's claim
   (see Section 4.4), and fixed a stale numeric comment. **Self-inflicted
   bug caught before shipping**: a JSDoc type annotation
   `{{date: string, count: number}[]}` collided with Jinja2's `{{ }}`
   expression syntax and broke `GET /habits/settings` with a template
   syntax error — caught by re-running the full regression suite after the
   docs pass, not by inspection alone.
6. **Production deployment incident** (see Section 4.1 — the most important
   lesson from this entire migration): applying the delivered `models.py`
   and `main.py` as full-file replacements silently deleted a `JobScoutRunLog`
   class (and its import in `routers/job_config.py` broke) because the
   `models.py` snapshot this work was based on had already gone stale
   relative to the live file by the time the deliverable was applied.
   Recovered via `git checkout` + manual reapplication of only the specific
   diff lines.
7. **Post-deploy user bug report**: drag-and-drop reordering on
   `/habits/settings` visually "stranded" a habit's heatmap/inline-edit
   panels at their old position after the row was dragged elsewhere — same
   root cause as the inline-edit panels appearing detached from their row.
   Root cause: `habit_settings_row.html` rendered three **flat sibling**
   `<div>`s (row, edit-fields, heatmap-section) per habit, but the
   drag-and-drop JS only ever grabbed and moved `.habit-settings-row`,
   leaving its siblings behind. Fixed by wrapping all three in one
   `.habit-settings-item` container that is the actual draggable unit and
   HTMX swap target; verified via an actual jsdom simulation (not just
   diff review) run against both the pre-fix and post-fix code to confirm
   the bug reproduced and then was resolved.
8. **Confirmed working in production** via live server logs (clean `200 OK`
   on every habits endpoint, including the duplicate-log and
   tap-refresh-tap scenarios that previously crashed) and direct user
   testing of drag/edit/heatmap interactions.

---

## 3. What Went Well

- The domain-folder pattern itself worked exactly as designed: zero other
  domains broke, confirmed via a live cross-domain consumer
  (`routers/dashboard.py`, which reads `Habit`/`HabitLog` via the top-level
  `from models import ...` re-export shim) returning correct data
  throughout every stage of this migration.
- `Base.metadata` stayed structurally sound throughout — same table count,
  no duplicate-registration errors, confirmed via identity checks
  (`models.Habit is domains.habits.models.Habit`) at every step, not just
  at the end.
- Every fix was verified against a **real running FastAPI instance**
  (`TestClient`/`httpx.AsyncClient` against the actual app, not just
  `py_compile` or manual reading) before being called done. This caught
  real bugs — including one I introduced myself during the docs pass — that
  static review would have missed.
- The final drag-and-drop fix was verified with an honest, adversarial
  method: the exact same jsdom simulation was run against the *original*
  buggy code first, to prove the test methodology actually catches the
  reported symptom, before trusting it to confirm the fix.

---

## 4. What Went Wrong — Lessons Learned (read this before starting the next domain)

### 4.1 — CRITICAL: never deliver a full-file replacement for a file you only saw a point-in-time snapshot of

This caused a real production incident (`JobScoutRunLog` silently deleted,
web container crash-looping). The mechanism: `models.py` and `main.py` are
**shared, actively-edited files** that every domain's migration touches in a
small, targeted way — but they were delivered as complete file replacements,
rebuilt from whatever snapshot was pasted into the work order at the start.
If the live file had changed *at all* since that snapshot (new classes, new
routers, anything) — which is likely, since these are actively-developed
files other work streams also touch — applying the delivered file wholesale
silently reverts/deletes that intervening work with no error, no warning,
until something downstream fails to import.

**Rule for every future domain migration:** for any file that is not brand
new (i.e., anything under the "Files edited" category, especially
`models.py` and `main.py`), deliver **exact before/after text blocks for
each specific edit**, not a full-file replacement. Whoever applies the
change should apply it as a targeted patch against the *current* live file,
ideally after confirming via `git diff` (or equivalent) that no unexpected
drift exists. Only genuinely new files (new domain's `models.py`, new
routers, new templates) are safe to deliver as complete files, because there
is no existing live version to accidentally clobber.

### 4.2 — Async SQLAlchemy: never touch an ORM object's attributes after a `commit()` or `rollback()` without an explicit re-fetch

The `greenlet_spawn has not been called` bug (Section 2, item 4) is not
habits-specific — it's a general anti-pattern that likely exists in **every
other router in this codebase**, since they all appear to follow the same
copy-paste structure (fetch an object via `await db.get(...)`, mutate/insert
something else, `await db.commit()`, then keep using the original object).
Whether this actually crashes depends on the real `database.py`'s
`expire_on_commit` setting, which nobody migrating a domain from outside the
container can see — so **assume the worst case (`expire_on_commit=True`,
SQLAlchemy's own default) and audit every router for this pattern as part of
each domain's migration**, not just habits'. The fix pattern is always the
same: re-fetch (`await db.get(Model, id)`) immediately before the object is
used again, after any `commit()`/`rollback()` in between.

### 4.3 — Template "context contracts" drift silently; document them, and verify the documentation itself

`habit_card.html` had a doc comment (twice, in fact — a duplicate) claiming
the template's context was `habit, today_logged (bool), streak (int), today`
— flat variables. The template body actually used `view.today_logged` /
`view.streak` (nested) and `habit.id` / `habit.color` (flat) — a completely
different shape. The stale comment didn't just fail to help; it **actively
documented the bug**, since it matched the buggy router code's old shape,
not the template's actual, already-updated body. For every domain's
partials: (a) write down the exact required context variables at the top of
the template, (b) grep the template body for every variable reference and
confirm the comment matches *exactly*, not approximately, and (c) treat any
mismatch between a doc comment and the actual body as a signal that
something drifted and is worth investigating, not just a docs nit.

### 4.4 — "Enforced at the DB level" comments are not enforcement; verify and declare

`HabitLog`'s docstring claimed a unique constraint on `(habit_id,
logged_date)` was "enforced at the DB level," but nothing in the SQLAlchemy
model declared it — meaning `Base.metadata.create_all()` (used by fresh dev
databases or test harnesses) would never actually create it, and anyone
reading the ORM model alone would have no way to know the constraint was
supposed to exist. It turned out, on checking the real production MariaDB
via `SHOW INDEX FROM habit_logs`, that the constraint genuinely did already
exist there (lucky — the alternative, an actual gap in production, was
equally possible and would have meant duplicate rows were silently
accumulating). **For every domain migrated, grep model docstrings/comments
for "enforced at," "constraint," "unique," "assumed," etc., and cross-check
each claim against the real production database** (`SHOW INDEX FROM
<table>` for MariaDB, `SHOW CREATE TABLE <table>` also useful) rather than
trusting the comment. Declare anything confirmed real explicitly in the
ORM model (`UniqueConstraint(...)` in `__table_args__`), with a comment
noting whether it was already live in production or needs an actual
migration.

### 4.5 — Multi-element list items need a single wrapper for drag-and-drop (and for any HTMX outerHTML swap) to move/replace them as a unit

Any settings/list UI in another domain that (a) supports drag-and-drop
reordering, and (b) renders more than one sibling element per list item
(e.g., a row plus a collapsible detail panel, plus a lazy-loaded chart/graph
section) is at risk of the exact same bug: JS that grabs only
`.closest('.the-row-class')` will move the row and leave its siblings
behind. **Check every domain's settings/list templates for this pattern
before migrating them** — if a list item renders N sibling top-level
elements, they need a wrapping container that is both the actual
`draggable="true"` element and the actual `hx-target` for any partial-swap
endpoint touching that item, with the JS drag logic and any
`document.getElementById(...).outerHTML = ...` calls all operating on the
wrapper, not the inner row.

### 4.6 — Starlette matches `Mount` routes in registration order — specific paths must be registered before general ones

Registering `app.mount("/static", ...)` before `app.mount("/static/habits",
...)` silently 404s every request under `/static/habits/*`, because
Starlette's router matches mounts in the order they were added and the
general one wins first. **Every future domain's static mount must be
registered before the general `/static` mount in `main.py`.**

### 4.7 — Re-running the full regression suite after *any* change (including "just docs") is mandatory, not optional

The Jinja2/JSDoc `{{ }}` collision (Section 2, item 5) was introduced during
a documentation-only pass with no intended behavior change, and it broke a
page outright. A change that "shouldn't affect behavior" is exactly the
kind of change most likely to be skipped from verification — don't skip it.

---

## 5. Checklist for Each Remaining Domain Migration

Follow this for every subsequent domain (recipes, workout, media, journal,
finance, jobs, code-intelligence, weekly-plan, etc.), in addition to
whatever domain-specific work order scopes the actual file moves:

1. **Scope exactly which model classes, router(s), templates, and static
   assets belong to this domain** — mirror the habits pilot's target
   structure (`domains/<name>/models.py`, `domains/<name>/routers/`,
   `domains/<name>/templates/`, `domains/<name>/static/`).
2. **Before editing `models.py` or `main.py`**, get the actual current
   content of those files (not a possibly-stale snapshot) — ideally
   `git diff`-verified immediately before applying anything, per Section
   4.1. Never assume a previously-seen copy is still accurate.
3. **Grep the entire real codebase** (not just what's been shared) for
   `from models import` and `from routers import <this domain's router
   name>` to find every cross-domain consumer that will need the
   re-export shim, or will need direct updating later.
4. **Audit this domain's router(s) for the async-SQLAlchemy
   commit/rollback-then-reuse pattern** (Section 4.2) — fix defensively
   regardless of whether it's provably reachable in the current code path.
5. **Audit this domain's partial templates**: confirm every doc comment's
   claimed context variables exactly match what the template body
   references (Section 4.3); check for the multi-sibling drag-and-drop
   risk (Section 4.5) in any list/settings UI.
6. **Audit this domain's models for "enforced at the DB level"-style
   comments** and verify each against the real production database
   (Section 4.4) before declaring anything in the ORM.
7. **Add this domain's template directory to `core/templating.py`'s
   `ChoiceLoader`** so `TemplateResponse("some_template.html", ...)` calls
   keep resolving by bare filename.
8. **Register this domain's static mount in `main.py` before the general
   `/static` mount** (Section 4.6).
9. **Deliver `models.py`/`main.py` changes as exact before/after text
   blocks**, not full-file replacements (Section 4.1). Only genuinely new
   files may be delivered whole.
10. **Verify against a real running instance**: full endpoint regression
    (every route the domain owns, plus at least one cross-domain consumer
    if one exists), re-run **after every change including docs-only
    changes** (Section 4.7). Use jsdom (or equivalent) for any
    drag-and-drop/client-side DOM logic, ideally proving the test also
    fails against the pre-fix code before trusting it confirms a fix.
11. **After deployment**, get real container logs and, ideally, a manual
    click-through from whoever owns the app — log lines showing `200 OK`
    confirm the request succeeded, not that the response rendered
    correctly on screen (the drag-and-drop bug was invisible in logs).

---

## 6. Final Cleanup — Only Once EVERY Domain Has Been Migrated

**Do not do any of this per-domain.** Each of these steps is only safe once
every single domain's classes/routers/templates have moved out of the
shared top-level files — doing it early would break every domain that
hasn't migrated yet.

1. **Remove every re-export shim from `models.py`.** Each domain's
   migration leaves behind a block like:
   ```python
   # TODO: remove after all cross-references are updated
   from domains.habits.models import Habit, HabitLog, HabitSettings
   from domains.habits.models import HabitCreate, HabitUpdate, HabitResponse, HabitLogResponse
   ```
   Before deleting each one: re-run the cross-codebase grep from step 3 of
   Section 5 for **every** migrated domain's class names, one final time,
   and confirm every remaining consumer has been updated to import from
   `domains.<name>.models` directly. Only then delete the shim block(s).
   Do this for all domains in the same pass, not incrementally, to avoid
   leaving the app in a state where some shims are gone and others aren't
   for no principled reason.
2. **Decide the fate of `models.py` itself.** Once every class has moved
   out and every shim is removed, `models.py` should contain nothing except
   possibly `from core.base_model import Base` for extremely stubborn
   legacy imports. At that point, either (a) delete it and update the
   handful of remaining `from models import Base` references to
   `from core.base_model import Base`, or (b) leave it as a single-line
   compatibility file indefinitely if some external/legacy consumer can't
   be updated. Prefer (a) — full removal — unless something concrete blocks
   it.
3. **Decide the fate of the top-level `routers/` folder.** Once every
   router has moved to `domains/<name>/routers/`, the old `routers/`
   directory should be empty and can be deleted, along with its
   corresponding `from routers import ...` lines in `main.py` (each
   already replaced per-domain during migration per Section 5, step 9).
4. **Decide the fate of the top-level `templates/` and `static/`
   directories.** These should end up containing only genuinely
   cross-domain/shared assets: `base.html`, `404.html`, `500.html`, any
   shared partials actually used by more than one domain, and global CSS/JS
   (`base.css`, `mobile.css`, `base.js`, etc.). Confirm nothing
   domain-specific still lives there before considering the migration
   series complete.
5. **Review and simplify `core/templating.py`'s `ChoiceLoader`.** By the
   end, it will have one `FileSystemLoader` entry per domain plus the root.
   Confirm the root entry is still needed (it should be, for the shared
   templates from step 4) and that the loader order still makes sense.
6. **Full schema/constraint audit.** Each domain's migration may have
   surfaced its own version of the "comment claims a DB constraint that
   isn't declared in the ORM" gap (Section 4.4). Once all domains are done,
   do one consolidated pass: for every table, compare `SHOW CREATE TABLE`
   (or equivalent) against what the ORM models declare, and reconcile any
   remaining drift — in whichever direction is correct (declare it in the
   model if it's real in the DB; add a migration if the model is right and
   the DB is missing it).
7. **Full regression suite, whole app.** Repeat the same style of
   verification used for habits (Section 5, step 10) but for literally
   every route in the app, not just one domain — this is the point where
   integration bugs between domains (if any) are most likely to surface,
   since it's the first time everything has moved simultaneously.
8. **Confirm `Base.metadata` integrity end-to-end**: same table count as
   before the entire migration series began, no duplicate-registration
   errors, and every table's columns/constraints unchanged (structurally
   verified, not just "no errors on import").
9. **Update or remove any remaining temporary scaffolding** created during
   the migration series — verification-only stub files, `TODO: remove
   after...` comments, this document itself (fold its still-relevant
   parts, if any, into permanent project documentation; the rest can be
   archived once the series is complete).
10. **Write a short closing summary** (this document's true final
    successor) confirming the domains/ restructuring is complete, what the
    final directory structure looks like, and pointing at wherever
    permanent architectural documentation for the project now lives.

---

## 7. Reference — Files Touched in This Migration (habits domain, final state)

**New:**
- `core/__init__.py`, `core/base_model.py`, `core/templating.py`
- `domains/__init__.py`, `domains/habits/__init__.py`
- `domains/habits/models.py`
- `domains/habits/routers/__init__.py`, `domains/habits/routers/habits.py`
  (moved from `routers/habits.py`, then edited in place across three
  separate fixes)
- `domains/habits/templates/habits.html`, `habits_settings.html`,
  `partials/habit_card.html`, `partials/habit_progress.html`,
  `partials/habit_settings_list.html`, `partials/habit_settings_row.html`
  (all moved from `templates/`, with `habit_card.html`,
  `habits_settings.html`, and `habit_settings_row.html` subsequently edited
  in place)
- `domains/habits/static/css/habits.css` (moved from `static/css/habits.css`,
  subsequently edited in place)

**Edited in place (shared files — apply as targeted diffs, see Section
4.1):**
- `models.py` — `Base` re-pointed to `core.base_model`; Habit-related
  classes replaced with a re-export shim
- `main.py` — habits router import re-pointed; new `/static/habits` mount
  added *before* the general `/static` mount

**Endpoints covered by the regression suite used throughout (reuse this
list's shape for each new domain):**
`GET /habits`, `GET /habits/settings`, `POST /habits/new`,
`POST /habits/log`, `DELETE /habits/log`, `GET /habits/progress`,
`GET /habits/heatmap/{id}`, `PATCH /habits/settings/grace-period`,
`PATCH /habits/reorder`, `PATCH /habits/{id}`, `DELETE /habits/{id}`,
plus one cross-domain consumer (`GET /dashboard`) and the domain's static
asset (`GET /static/habits/css/habits.css`).
