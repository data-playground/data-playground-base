# Life OS — Project Governance Bible

**Status:** Living document. Last updated after Work Orders #1–4 (habits,
blog+code_intel, jobs, explorer domain migrations). Individual sections
carry their own "Status" notes as they're superseded by later work — check
those before trusting this front-matter line for any one section.

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
  SQL helpers. DAG files stay under `airflow/dags/`, organized into
  per-domain subfolders (`airflow/dags/<domain>/`) rather than relocated
  into `domains/*/dags/` — see §2.5, which documents this as the settled,
  permanent approach (not a deferred step, per WO#18).

### 2.3 AI Service Layer (Target State — In Progress)
**Current state (as of this document):** six independent implementations of
"call an LLM provider" exist across `blog_agents.py`, `recipe_agents.py`,
`weekly_agents.py`, `gemini_client.py`, `workout_plans.py` (inline), and
`media_recommend.py` (inline), plus `finance_upload.py` using the
`google-genai` SDK directly. This is tracked technical debt, not yet
resolved by the domain-folder migrations (WO#1–4 intentionally left
`blog_agents.py` untouched — see WO#2's hard boundaries).

> **Known staleness, not yet corrected:** the paragraph above no longer
> reflects reality. WO#11–16 have since migrated `job_agents.py`,
> `recipe_agents.py`, `weekly_agents.py`, `workout_plan_ai_generator.py`
> (the renamed successor to `workout_plans.py`'s AI-generation half), and
> `media_recommend.py`, plus all three of `blog_agents.py`'s provider
> implementations (Gemini, Groq, Cerebras), into `services/ai/providers/`.
> `finance_upload.py`'s SDK-based call has been explicitly documented as a
> deliberate, permanent exception rather than remaining debt (see
> `services/ai/README.md`'s "SDK Exceptions" section, added in WO#16).
> This "Current state" section is left unedited for now — pending a
> rewrite once WO#16's own open items (verifying `services/ai/base.py`,
> `keys.py`, and `__init__.py` against real source, and resolving the
> "five vs. six implementations" discrepancy first flagged in WO#11's
> postmortem) are closed, so it only needs rewriting once rather than
> twice.

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

**Status: historical/closed.** WO#20 removed every remaining domain shim
from root `models.py`, and WO#22 removed root `models.py` itself once a
repo-wide grep confirmed it had no real consumers left — the
mapper-registration guarantee this section originally relied on shims for
now lives as an explicit import block in `database.py`. This section is
kept for historical context; no further shim-removal work is expected.

### 2.5 DAG Organization — Resolved (see WO#18)
Earlier versions of this document assumed relocating `airflow/dags/*.py`
into domain-scoped subfolders would require a coordinated
`docker-compose.yml` volume-mount change and carried real risk of DAGs
failing to schedule silently. That assessment has been superseded by
actual execution.

**What's actually true, confirmed by WO#18:**
- Airflow's DAG discovery recursively scans the configured `dags_folder`
  for `.py` files containing DAG objects — subfolder depth doesn't matter.
- `docker-compose.yml` already mounts `./airflow/dags:/opt/airflow/dags`
  as a whole tree, so organizing into subfolders *within* that
  already-mounted path requires **no volume-mount change** (confirmed via
  an empty `docker-compose.yml` diff across the relocation).
- Every DAG file resolves its own imports via absolute container paths
  (`sys.path.insert(0, '/opt/airflow/project')`, `sys.path.insert(0,
  '/opt/airflow/project/airflow')`), not paths relative to the DAG file's
  own location — moving the file doesn't touch these.
- Every DAG's `dag_id` is an explicit string literal, not derived from
  file path — the scheduler, UI, run history, and
  `services/airflow_service.py`'s `trigger_airflow(dag_id, ...)` helper
  are all keyed on this string and are unaffected by relocation.

**Current state:** 13 DAG files now live under `airflow/dags/<domain>/`
subfolders (`blog/`, `code_intel/`, `jobs/`, `journal/`, `media/`). Two
DAG files — `life_os_staging_promoter.py` and
`life_os_refresh_streaming_availability.py` — remain at the flat
`airflow/dags/` root by deliberate choice, not oversight (they postdate
WO#18's own file list and were never in its scope). This was a pure
relocation for the 13 that moved — zero code changes inside any DAG file,
zero `docker-compose.yml` changes. See WO#18 and its postmortem for the
full verification (byte-identical file diffs, `git log --follow` history
preservation).

**What this does NOT resolve:** moving the *agent modules* under
`airflow/agents/*.py` into a similar structure is a different,
higher-risk question. Several of those files (`recipe_agents.py`,
`weekly_agents.py`, `blog_agents.py`) are imported directly by FastAPI
routers as well as (or instead of) DAGs — unlike DAG files, which nothing
outside the Airflow layer imports — so the "pure relocation, zero code
change" property does not automatically transfer. Don't assume it does;
redo the same discovery/import/identity analysis WO#18 performed for
DAGs, applied fresh to `agents/*.py`, before scoping that move.

**Update (WO#31):** the DAG-only tier of `airflow/agents/*.py` has since
been relocated too — see `airflow/agents/jobs/` and `airflow/agents/media/`.
Unlike the DAG move above, this was *not* a zero-code-change relocation:
every consumer's import line had to change in the same commit, since
nothing about a Python module's own import path is location-independent
the way a DAG's `dag_id` is. The remaining two tiers (router-only modules
arguably misplaced under `airflow/agents/` at all, and `blog_agents.py`,
which is consumed by both DAGs and routers simultaneously and can't move
incrementally) remain deliberately unscoped — see WO#31's own postmortem
and the Track D scoping analysis before attempting either.

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

### 2.7 Secrets & Environment Variable Wiring

**Rule: when a router, service, or agent module needs a secret or
environment variable to talk to another internal service (Airflow, a
database credential) or an external one (an LLM provider, a third-party
API), verify that variable is actually passed through in *every*
`docker-compose.yml` service that will call it at runtime — not just the
service that already owns that secret's other configuration values.**

This is not a hypothetical risk; it's what actually happened. WO#35 (the
`medium` domain build) found that `docker-compose.yml`'s `web` service was
missing `AIRFLOW_SECRET_KEY` — present for the three `airflow-*` services,
where it's used to set Airflow's own admin password and Flask secret key,
but never passed through to `web`, where `services/airflow_service.py`
actually needs it to authenticate outbound `trigger_airflow()` calls. This
silently broke **every** "click a button, fire a DAG" feature already
shipped in this app — blog's Scout/Creator/Finalizer/Idea Expander,
code_intel's README Writer and Narrate/Comment/Improve DAGs — not just the
medium domain being built at the time. It went undetected because nothing
had previously exercised that exact code path end-to-end from the `web`
container; medium's build was simply the first feature to do so, and only
found it because its own acceptance criteria required confirming the
trigger actually reached Airflow rather than accepting a mocked response.

**Going forward:**
- Before treating any new Airflow-trigger, external-API, or cross-service
  integration as "done," confirm the credential/secret it depends on is
  present in the environment block of every `docker-compose.yml` service
  that will actually invoke it — not just the service that defines or
  owns the secret.
- Do not assume an existing, already-working integration (e.g. another
  domain's DAG trigger) proves a given secret is correctly wired
  everywhere it's needed. As this incident shows, a missing var on one
  service can sit undetected indefinitely if nothing happens to exercise
  that specific service-to-service path.
- When a work order's acceptance criteria include "reaches the DAG/API
  correctly," a mocked or stubbed call — which most work orders in this
  program have had to rely on for lack of live infrastructure access — is
  not sufficient to catch this class of bug. Mark that criterion ⚠️ rather
  than ✅ unless the live credential path was actually exercised, the same
  way any other unverifiable-in-sandbox criterion is handled per §4.3's
  standing template.

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

  **Status: resolved (see WO#17).** Every page identified above, plus
  eleven more found in a subsequent full-tree audit, has been migrated
  onto the shared `#toast`/`showToast()`. Toast display duration was
  harmonized to 2600ms across all pages as a disclosed side effect. Kept
  here for historical context, not as an open item.
- **Sidebar/theme JS:** `sidebar_js.html` re-implements functions already
  in `base.js`. New pages should include `base.js` via `base.html` and
  never re-declare `setTheme()`/`toggleSidebar()`/mobile handlers locally.

  **Status: resolved (see WO#17).** `sidebar_js.html` had zero remaining
  includes anywhere in the codebase and was deleted.
- **Inline `style="..."` attributes:** dominant across most templates.
  Rule going forward: any inline style pattern repeated 3+ times within a
  single template, or matching an existing `base.css` primitive
  (`.panel`, `.badge`, `.btn`, `.stat-card` equivalents), must use the
  class instead of a copy-pasted inline style. This is not retroactively
  enforced on existing templates as part of routine migrations (that would
  balloon every work order's scope) — it's enforced on new/edited code.
- **Multiple AI client implementations:** see §2.3.

### 3.3 Migration Debt Tracker

**Status: historical/closed.** Every domain originally named in this
tracker has since been migrated: `finance` (WO#5), `journal` (WO#6),
`recipes` + `pantry` (WO#7), `workout` (WO#8), `media` (WO#9), and
`planning` (`weekly_plan` + `intent`, WO#10). Combined with the four
priority domains migrated earlier (`habits` WO#1, `blog` + `code_intel`
WO#2, `jobs` WO#3, `explorer` WO#4), every domain on the original backlog
now lives under `domains/<name>/`. `dashboard` remains, intentionally, the
one domain-less top-level router — see §2.2.

This section was found stale — still listing all six domains above as
un-migrated well after their work orders had actually shipped —
independently by four separate postmortems (WO#25, #26, #28, #29), each
re-deriving the same "wait, is this actually done?" check on its own
rather than trusting this document. It's corrected here for the same
reason §2.4 was: a tracker that isn't updated the moment its own condition
is satisfied becomes actively misleading rather than merely outdated. If a
domain is ever un-migrated, split further, or newly created, add it here
explicitly rather than assuming this section stays silently accurate on
its own.

**Note on a related but distinct concern:** this tracker covers
domain-*folder* migration status only. Router *file-size* compliance
within an already-migrated domain is a separate axis (§1.2) with its own
tracking — see the Router Line-Limit Remediation series (WO#19,
WO#23–30) for that status, not this section.

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

**Candidate additions flagged by later work orders, not yet folded into
the template above — track these before the next major redraft:**
- Any work order that splits a router into multiple files should require,
  as its own Step 0, either listing `main.py` in SCOPE or explicitly
  confirming every resulting router's registration — several Track C
  postmortems (WO#28, #29, #30) independently found or narrowly avoided a
  broken registration because this wasn't a mandatory check (WO#30 in
  particular found and had to fix a real, pre-existing 404 this way).
- A router split that separates literal-path routes from a parametric
  `/{id}` route into different files should require an explicit check for
  shadowing (not just exact-path collision) and, ideally, an automated
  route-order regression test — per WO#23's finding for `habits`/
  `habits_settings`.
- A work order whose STEPS pins a closed set of functions to remain in a
  file *and* whose ACCEPTANCE CRITERIA requires that file to hit a line
  ceiling should have its author pre-check that the arithmetic actually
  works (count the pinned functions' lines before publishing the work
  order) — this exact conflict produced open items in both WO#24 and
  WO#28.

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

This rule is about *not bundling*, not about *never fixing quickly* — a
high-severity, directly-adjacent bug found while a file is already open
for a required edit can still be fixed in the same session, but only if
it's kept as a clearly labeled, separately callable-out change within the
diff (not silently merged into the migration's own hunks), with its own
line in whatever change-tracking the repo uses. WO#30's handling of a
pre-existing `main.py` router-registration bug it found while doing its
own required registration edit is the reference example for how to do
this correctly.

### 4.6 What "Done" Means for a Domain Migration
A domain is considered migrated when:
- Its `models.py`, routers, templates, and static assets all live under
  `domains/<name>/`.
- `main.py` and `core/templating.py` reference the new paths.
- A legacy shim exists in root `models.py` for any external consumer
  (normally just `dashboard.py`) — **historical note:** per §2.4, this no
  longer applies to any currently-migrated domain, since shims and root
  `models.py` itself have both been fully retired (WO#20, WO#22). This
  criterion is kept for any future domain migration, should one occur.
- All acceptance criteria in its work order passed (✅ or an explained ⚠️).
- No unrelated behavior changed (confirmed via the `Base.metadata`
  identity check pattern established in WO#1, and functional
  re-verification of every affected endpoint).

It is *not* considered done if "cleanup" happened alongside it (dead code
removal, style fixes, bug fixes) — those are separate, separately
reviewable changes by design (§3.2, §4.5).

---

## 5. Open Items Tracked for Later (Not Blocking Current Work)

These are recorded here so they aren't lost, per standing practice:

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
4. **New domains** (sports data, Medium article extraction, etc.) — the
   backlog this was originally written against is now substantially
   cleared: Medium (WO#35) has shipped as its own domain under
   `domains/medium/`, and NBA/Soccer data domains are in progress (see the
   project's own work-order index). New domains going forward should be
   built directly inside the `domains/` structure from day one, per the
   pattern these three establish, rather than needing a later migration.

---

## 6. Amendment Process
This document is updated whenever a work order surfaces a rule worth
generalizing (as happened repeatedly during WO#1's review — the mount
ordering fix, the pre-existing-bug handling rule, and the substitute-
verification rule all originated as one-off corrections and were promoted
into standing rules here). Treat every work-order report's "Notes" section
as a candidate source of the next amendment, not just a log.

**Amendment log (sections touched, most recent first):**
- §3.3, §2.7 (new) — corrected the Migration Debt Tracker to
  historical/closed after four independent postmortems (WO#25, #26, #28,
  #29) caught it listing already-migrated domains as outstanding; added a
  new rule on threading secrets through every `docker-compose.yml` service
  that needs them, after WO#35 found `AIRFLOW_SECRET_KEY` missing from the
  `web` service and traced the resulting silent breakage across multiple
  domains' Airflow-trigger features. §3.2's toast/sidebar-JS entries also
  marked resolved per WO#17.
- §2.4 — marked historical/closed after WO#20 (shim removal) and WO#22
  (root `models.py` deletion, `configure_mappers()` verification attempt).
- §2.5 — rewritten after WO#18 (DAG reorganization) replaced the original,
  more cautious assumption with confirmed, lower-risk reality; later noted
  the WO#31 Tier 1 agent-module reorg as a related but distinct case.
- §2.2 — one stale cross-reference to §2.5 corrected during WO#21's
  authorized follow-up.
