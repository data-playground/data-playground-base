# Life OS — Project Governance Bible

**Status:** Living document. Last updated after Work Orders #1–4 (habits,
blog+code_intel, jobs, explorer domain migrations).

This document is the permanent law for all development on this repository —
human or AI-assisted. Any future coding session, work order, or ad hoc change
should be checked against this document before, not after, the work happens.

---

## 1. Coding Standards & Style Guide

### 1.1 Naming Conventions
- **Routers:** `snake_case.py`, named after the domain or sub-feature they
  own (`ci_files.py`, not `code_intel_files_router.py`). Prefix shared
  route groups clearly when a domain has multiple routers (`ci_*` for
  code_intel, `job_*`/`ats`/`staging` for jobs).
- **Models:** PascalCase classes, one `models.py` per domain (see §2.1).
  Enums use PascalCase with UPPER_SNAKE_CASE members, matching the existing
  codebase convention (`BlogIdeaStatus.IDEA_GENERATED`).
- **Templates:** `snake_case.html`. Partials always live under a
  `partials/` subfolder within their domain's `templates/` directory, never
  loose alongside full-page templates.
- **Static assets:** mirror the domain name in both the folder path and the
  URL mount (`domains/jobs/static/css/jobs.css` served at
  `/static/jobs/css/jobs.css`). Never let a static filename collide across
  domains — the domain-scoped mount makes this a non-issue going forward,
  but keep it in mind if a shared asset is ever needed (put it in
  `shared/static/` instead, not in any one domain).

### 1.2 File Size Limits
- **Routers: 300 lines, hard ceiling.** This was already a stated rule in
  `CONTRIBUTING.md` before this governance pass but was not enforced —
  `weekly_plan.py`, `media_recommend.py`, `workout_plans.py`, and
  `ci_readme.py` all exceeded it at time of writing. Going forward:
  - Any router approaching 250 lines should be split by responsibility
    (CRUD vs. AI-generation, e.g. `workout_plans.py` →
    `workout_plans_crud.py` + `workout_plan_ai_generator.py`) *before* it
    crosses 300, not after.
  - This should be a CI-checkable lint step (line count per file under
    `domains/*/routers/`), not just a reviewer's judgment call.
- **Models:** no hard line limit per domain `models.py`, since domain size
  varies naturally, but if a single domain's model file exceeds ~400 lines,
  consider whether it's actually two domains that were merged prematurely.
- **Agent/service files:** no hard limit, but a file mixing more than one
  provider's raw API logic (as `blog_agents.py` currently does) is a signal
  it should be decomposed per the AI service layer plan (§2.3).

### 1.3 Formatting
- Follow existing patterns already dominant in the codebase: `# ── SECTION
  NAME ──────────` style dividers within Python files, Google-style
  docstrings with Args/Returns/Raises for non-trivial functions.
- Jinja2 partials should open with a comment block stating: what
  endpoint(s) return them, what context variables they expect, and what
  they're swapped into (this convention is already followed well in most
  existing partials — keep it universal).

---

## 2. Architecture Rules

### 2.1 Domain-Folder Structure (Mandatory)
Every feature domain lives under `domains/<name>/` and owns:
```
domains/<name>/
    __init__.py
    models.py              # this domain's ORM + Pydantic classes only
    routers/
        __init__.py
        <router files>.py
    templates/
        <page>.html
        partials/
            <fragment>.html
    static/
        css/
        js/
```
Not every domain needs every subfolder (e.g. `explorer` has no `models.py`,
`code_intel` has no dedicated CSS) — omit what isn't needed rather than
creating empty placeholders.

**Rule: all AI integration logic must live in `services/ai/`, never inline
in a router, template, or DAG.** (See §2.3 — this is not yet fully true
project-wide as of this document's writing and is tracked as outstanding
work, but it is binding for all *new* code starting now.)

### 2.2 Cross-Domain Rules
- **A domain's `models.py` must not import another domain's `models.py`
  directly.** SQLAlchemy `relationship()` calls that cross domain
  boundaries use **string class names** (e.g. `relationship("CodeFile",
  ...)`), which resolve via the shared mapper registry at query time, not
  import time. This is already the pattern used for `BlogIdea.code_file` /
  `CodeFile.blog_ideas` and requires no special import gymnastics — just
  make sure both domains' `models.py` get imported somewhere before the
  first query runs (see §2.4's shim mechanism).
- **Domains with a live FK relationship must be migrated together**, in the
  same work order (precedent: `blog` + `code_intel` in WO#2). Splitting
  them across separate work orders creates a window where one domain
  references a not-yet-relocated class.
- **`routers/dashboard.py` is the one sanctioned cross-domain reader.** It
  is allowed to import from any domain's `models.py` for read-only summary
  purposes. No other router should import another domain's models directly
  — if two domains need to share data, that's a signal either (a) the data
  belongs in a shared/core location, or (b) the two domains should be
  merged, or (c) the interaction should go through an HTTP/service call,
  not a direct model import.
- **DAGs never import `models.py`, `database.py`, or any router/service.**
  This rule predates this governance pass (`CONTRIBUTING.md`) and remains
  absolute. All DAG database access goes through `airflow/dag_db.py` raw
  SQL helpers. DAG files stay in `airflow/dags/` (not yet relocated into
  `domains/*/dags/` as of this writing — see §2.5 for why that move is
  deliberately deferred).

### 2.3 AI Service Layer (Target State — In Progress)
**Current state (as of this document):** six independent implementations of
"call an LLM provider" exist across `blog_agents.py`, `recipe_agents.py`,
`weekly_agents.py`, `gemini_client.py`, `workout_plans.py` (inline), and
`media_recommend.py` (inline), plus `finance_upload.py` using the
`google-genai` SDK directly. This is tracked technical debt, not yet
resolved by the domain-folder migrations (WO#1–4 intentionally left
`blog_agents.py` untouched — see WO#2's hard boundaries).

**Target state:**
```
services/ai/
    __init__.py          # public exports: call_ai_text(), call_ai_json()
    base.py               # shared retry/backoff, shared exceptions
    providers/
        gemini.py
        groq.py
        cerebras.py
    keys.py               # single get_provider_key(provider)
    README.md             # model-routing rationale — moved from blog_agents.py's
                           # header comment, since it applies project-wide
```
**Rule for all new code:** any new AI provider call must go through this
layer once it exists. Until it exists, do not add a seventh independent
implementation — extend one of the existing ones and flag the duplication
in a code comment rather than compounding it.

### 2.4 Legacy Import Shims
Every domain migration leaves a temporary re-export shim in the root
`models.py`:
```python
# TODO: remove after all cross-references are updated
from domains.<name>.models import ClassA, ClassB, ...
```
**Rule:** these shims are scaffolding, not permanent architecture. Once a
domain's only remaining external consumer is `routers/dashboard.py` (the
sanctioned cross-domain reader), update `dashboard.py` to import directly
from `domains.<name>.models` and delete that domain's shim. Track shim
removal as its own small cleanup task per domain, not bundled into the
migration work order itself (this keeps each migration's diff focused and
its acceptance criteria clean).

### 2.5 Why DAGs Haven't Moved Yet
DAG relocation (`airflow/dags/*.py` → `domains/*/dags/`) is deliberately
**out of scope** for every migration work order so far. It requires a
coordinated `docker-compose.yml` volume-mount change (the Airflow
containers currently mount `./airflow/dags` directly), and getting that
wrong breaks DAG scheduling silently rather than failing loudly like a
FastAPI import error would. This is tracked as a distinct, later phase —
do not fold it into a routine domain migration work order.

### 2.6 Templating & Static Serving
- **Templates:** `core/templating.py` holds one shared `Jinja2Templates`
  instance using a `jinja2.ChoiceLoader` that searches the root
  `templates/` directory first, then each `domains/*/templates/` in the
  order they were added. Routers always call
  `templates.TemplateResponse("some_file.html", ...)` with just the
  filename — never a path — so the loader can find it regardless of which
  physical folder it lives in. Every new domain migration adds its
  `templates/` root to this `ChoiceLoader` list.
- **Static assets:** each domain gets its own `StaticFiles` mount
  (`/static/<domain>`), registered **before** the general `/static` mount
  in `main.py`. This ordering is not cosmetic — Starlette matches `Mount`
  routes in registration order, and a general `/static` mount registered
  first will silently 404 every domain-specific static request before the
  more specific mount ever gets a chance (this was a real bug caught during
  WO#1 and is now a standing rule, not just a lesson).

---

## 3. DRY & Consolidation Mandate

### 3.1 Before Writing New Code
Before adding a new utility function, template partial, CSS class, or
provider-call wrapper, check for an existing one in this order:
1. `shared/` (or currently, `base.css` / `base.js` at the static root) —
   is there already a primitive that does this?
2. The domain you're working in — does a sibling file already solve this?
3. Another domain's equivalent file — is there a pattern worth reusing
   (not necessarily importing, since cross-domain imports are restricted
   per §2.2, but worth matching the *shape* of)?

Only after checking all three should new code be written.

### 3.2 Known Duplication Still Being Worked Down
This section exists so future sessions don't "rediscover" the same debt
and treat it as new:
- **Toast notifications:** `base.js` defines the canonical `#toast` +
  `showToast()`. Several pages (`recipes.html`, `pantry.html`, `blog.html`,
  `jobs.html`, `workout.js`) still define local variants with differently
  named toast elements (`#recipe-toast`, `#pantry-toast`, etc.). Rule:
  any new page must use the shared `#toast`/`showToast()` from `base.js`.
  Existing duplicates get cleaned up opportunistically when their domain
  is next touched for any reason — not a standalone project.
- **Sidebar/theme JS:** `sidebar_js.html` re-implements functions already
  in `base.js`. New pages should include `base.js` via `base.html` and
  never re-declare `setTheme()`/`toggleSidebar()`/mobile handlers locally.
- **Inline `style="..."` attributes:** dominant across most templates.
  Rule going forward: any inline style pattern repeated 3+ times within a
  single template, or matching an existing `base.css` primitive
  (`.panel`, `.badge`, `.btn`, `.stat-card` equivalents), must use the
  class instead of a copy-pasted inline style. This is not retroactively
  enforced on existing templates as part of routine migrations (that would
  balloon every work order's scope) — it's enforced on new/edited code.
- **Multiple AI client implementations:** see §2.3.

### 3.3 Migration Debt Tracker
Domains not yet moved into the `domains/` structure, as of this document:
`finance`, `journal`, `recipes` (+ `pantry`), `workout`, `media`,
`planning` (`weekly_plan` + `intent`). `dashboard` intentionally stays at
the top level (see §2.2). These migrate on demand using the standing
work-order template in §4.3 — there is no forced deadline, since none of
them are on the priority list and all are "tested but not in heavy use."

---

## 4. AI Collaboration Guidelines

### 4.1 Scoping Principle
**Never hand an AI coding session the whole repository when the task is
domain-scoped.** The entire reason the `domains/` structure exists is to
make it possible to say "everything you need is in `domains/<name>/` plus
`services/`, `core/`, and `shared/templates/base.html` — do not touch
anything else" and have that be a true, complete, and safe instruction.
Every future refactor or feature request should be scoped this precisely
before it's handed off.

### 4.2 Role Framing Is Not Optional
Every work order given to an AI coding agent must open with an explicit
**ROLE** statement constraining its behavior — not just a task description.
A refactoring task and a greenfield feature task need different default
postures (minimal/reversible/verifiable vs. exploratory/creative), and an
agent left to infer this from context alone will drift toward "helpful
improvements" that make diffs harder to review and revert.

### 4.3 Standing Work-Order Template
Every migration or refactor work order must use this structure. This
template is the product of real corrections found during WO#1–4 (the
mount-ordering bug, the pre-existing-bug handling rule, the substitute-
verification rule) — don't simplify it away.

```markdown
## ROLE
[constrain the agent's behavior explicitly — e.g. "refactoring engineer,
not a feature builder"]

## HARD BOUNDARIES
- Only read/edit files explicitly listed in SCOPE.
- No schema/behavior changes unless the task is explicitly about that.
- If instructions conflict with actual code found, STOP and report —
  don't improvise.
- [task-specific exclusions, e.g. "do not move DAG files," "do not touch
  file X even though it's related"]

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
1. Do NOT fix — out of scope by default.
2. Reproduce against the pre-change baseline to confirm it's not a
   regression you introduced.
3. Report under Notes with enough detail to file a ticket.
4. Mark the related acceptance criterion ⚠️, not ❌, with a one-line
   explanation of the distinction.

## WORKING METHOD
Execute in order. Verify incrementally after behavior-changing steps, not
only at the end. If an acceptance criterion needs a resource not in SCOPE,
don't skip it silently — do the closest achievable check, state the
substitution explicitly, mark ⚠️.

## OUTPUT FORMAT
1. Files created
2. Files moved (old → new)
3. Files edited (path — description; flag anything beyond the literal
   instructions and explain why it was necessary)
4. Acceptance criteria results (✅/❌/⚠️ + one-line reason for non-✅)
5. Notes (improvements not acted on, pre-existing bugs found, risks)

## ROLLBACK
State the safe rollback method (usually: git checkout on every file in
sections 1–3 of the output).

## SCOPE
[explicit file list — include every config/resource file any acceptance
criterion depends on, so the agent isn't left guessing what's available]

## STEPS
[ordered, specific]

## ACCEPTANCE CRITERIA
[each one achievable with what's in SCOPE; pre-adjust criteria that would
otherwise depend on an out-of-scope resource, rather than leaving the gap
for the agent to discover mid-task]
```

### 4.4 Report Review Checklist
When a work-order report comes back, check in this order:
1. Did every HARD BOUNDARY get respected? (Check "Files edited" against
   the exclusion list explicitly, don't just skim.)
2. Are all ❌/⚠️ items genuinely out of the agent's control (missing
   resources, pre-existing bugs) rather than incomplete work?
3. Does the Notes section surface anything that needs its own ticket
   (per §4.5)? File it separately — don't let it get folded into a
   "while we're at it" fix on the next work order.
4. Only after 1–3: confirm the acceptance criteria that matter functionally
   actually passed.

### 4.5 Bugs Found During Migration Are Not Migration Work
If a work order's verification surfaces a genuine, pre-existing bug (see
example precedent: the `habits` log/unlog 500 error found during WO#1,
caused by a `**view` dict-spread mismatch with what `habit_card.html`
expected), that bug gets its own standalone ticket with its own fix and
its own verification — never bundled into the migration's diff, even if
the fix is one line. This keeps migration diffs reviewable as "pure
relocation" and keeps bug fixes independently revertable.

### 4.6 What "Done" Means for a Domain Migration
A domain is considered migrated when:
- Its `models.py`, routers, templates, and static assets all live under
  `domains/<name>/`.
- `main.py` and `core/templating.py` reference the new paths.
- A legacy shim exists in root `models.py` for any external consumer
  (normally just `dashboard.py`).
- All acceptance criteria in its work order passed (✅ or an explained ⚠️).
- No unrelated behavior changed (confirmed via the `Base.metadata`
  identity check pattern established in WO#1, and functional
  re-verification of every affected endpoint).

It is *not* considered done if "cleanup" happened alongside it (dead code
removal, style fixes, bug fixes) — those are separate, separately
reviewable changes by design (§3.2, §4.5).

---

## 5. Open Items Tracked for Later (Not Blocking Current Work)

These are recorded here so they aren't lost, per standing practice — they
are explicitly **not** being worked on until the domain migration backlog
(§3.3) is either complete or deliberately deprioritized further:

1. **Interaction tracking → adaptive Dashboard.** Needs a lightweight
   event-log table plus a template-variant system. Natural fit once
   `domains/dashboard/` exists as its own bounded space.
2. **Dashboard digest email.** The job-scout digest DAG
   (`life_os_daily_digest.py`) is the template to generalize. Needs a
   stable Dashboard data contract first, plus a new `services/email/`
   layer (HTML email templates are a different concern from web templates).
3. **In-Docker coding environments (Jupyter + browser IDE).** Infrastructure
   addition, not a FastAPI domain — belongs in `docker-compose.yml` +
   an `infra/` or `dev-tools/` folder. Needs its own memory/caching design.
4. **New domains** (sports data, Medium article extraction, etc.) — these
   are exactly what the domain-folder pattern was derisked for. Once the
   backlog in §3.3 is cleared, new domains should be built directly inside
   this structure from day one rather than needing a later migration.

---

## 6. Amendment Process
This document is updated whenever a work order surfaces a rule worth
generalizing (as happened repeatedly during WO#1's review — the mount
ordering fix, the pre-existing-bug handling rule, and the substitute-
verification rule all originated as one-off corrections and were promoted
into standing rules here). Treat every work-order report's "Notes" section
as a candidate source of the next amendment, not just a log.
