# Work Order #35 — Medium Article Extraction (Revisit Existing Implementation)

*Different in kind from WO#33/#34. This is not a greenfield build — an
implementation already exists somewhere and needs revisiting. Because the
starting shape is unknown until the code is actually seen, this work order
runs in two phases: assess first, get a direction confirmed, then build.
Don't collapse the two phases — the right architecture literally cannot be
decided until Phase 1's findings are in.*

---

## ROLE
You are modernizing an existing feature to match this codebase's current
architecture. Unlike WO#33/#34, you're not starting from nothing — treat
the existing implementation as real prior work to understand before
changing it, the same care a refactoring work order would take, but once
you understand it, you have real latitude on how to rebuild/integrate it —
this isn't a "preserve every line" migration either. It's a genuine
in-between: respect what exists, then modernize it with judgment.

## HARD BOUNDARIES
- Phase 1 (assessment) must complete and be reviewed before Phase 2 (build)
  starts — this is not optional, and not just a formality: the
  own-domain-vs-fold-into-`blog` decision genuinely changes the shape of
  everything in Phase 2.
- Whichever direction is chosen, any LLM/AI calls route through
  `services/ai/` — do not build a new independent provider-calling
  implementation, and do not assume the existing implementation's own
  AI-calling approach (if it has one) should be preserved as-is; it
  predates this codebase's AI Service Layer consolidation.
- If DAG-scheduled ingestion/extraction is part of the final design: it
  must never import `models.py`, `database.py`, or any router/service.
- If folded into `blog`: do not bypass the existing
  Researcher→Ghostwriter→Refiner→Editor pipeline shape without a clear
  reason — understand why it exists (see `blog_agents.py`'s own module
  docstring) before deciding whether Medium content needs something
  different.

## WORKING METHOD
Phase 1 is investigation and a written recommendation — no destructive
changes, no large rewrites, during this phase. Phase 2 only starts once
someone has actually looked at Phase 1's findings and confirmed a direction.

## OUTPUT FORMAT

**Phase 1 report:**
1. What the existing implementation actually does today (data flow, what's
   automated vs. manual, what output it produces)
2. How (if at all) it currently integrates with the rest of this app
3. What's broken, missing, or why it needs revisiting (per the owner's own
   framing, gathered in Step 1)
4. Recommendation: own domain (`domains/medium/`) vs. fold into `blog`'s
   existing AI pipeline — with reasoning, not just a preference
5. Open questions for the owner to resolve before Phase 2

**Phase 2 report** (once a direction is confirmed):
1. Files created / moved / edited
2. Design decisions made and why
3. Acceptance criteria results
4. Notes

## ROLLBACK
Phase 1: nothing to roll back (investigation only). Phase 2: `git checkout`
on everything created/edited in that phase.

---

## PHASE 1 — Request materials and assess

### Step 1 — Request materials
Acknowledge this work order, then request:
1. **The existing Medium article extraction implementation** — the actual
   code, wherever it lives (this repo, a separate script, a notebook), or a
   working description if the code itself isn't readily shareable.
2. **What's wrong with it or what's wanted differently** — is this a bug
   fix, a structural rebuild to match the domain-folder pattern, a feature
   extension, or some combination? Don't assume "needs modernizing" means
   the same thing the owner means by it.
3. **Any preference on own-domain vs. fold-into-blog**, even a loose one —
   useful context even though the final call should follow from what Phase
   1 actually finds, not be decided blind.

### Step 2 — Assess
Once the implementation is in hand:
1. Trace what it actually does end to end — where does input come from
   (a URL? a feed? manual paste?), what transformation/extraction happens,
   where does output go (does it produce a draft, a fully formatted post,
   raw text?).
2. Check whether it already touches anything in this codebase (does it
   write to a table? call an existing service? run as a cron job outside
   Airflow entirely?).
3. Form a recommendation on own-domain vs. fold-into-blog, reasoning from
   what you found — e.g., if the existing tool's output already resembles
   a blog-post draft, folding into `blog_agents.py`'s pipeline (feeding a
   Medium URL/export in as if it were `code_narrative`/`author_notes`
   context for the Ghostwriter, or as a new distinct agent function
   alongside the existing ones) may need less new code than a full separate
   domain. If it's structurally more like an independent content-management
   flow (its own review states, its own storage needs beyond what
   `blog_ideas` already models), a dedicated `domains/medium/` may fit
   better.
4. Write and share the Phase 1 report. **Stop here** and wait for the
   direction to be confirmed before writing any Phase 2 code.

### Phase 1 Acceptance Criteria
- [ ] Step 1 happened before any assessment was written.
- [ ] The report accurately describes what the existing implementation does
  today — verified by tracing its actual logic, not inferred from a
  description alone.
- [ ] A recommendation is made with explicit reasoning, not just asserted.
- [ ] No code was created, moved, or rewritten during this phase.

---

## PHASE 2 — Build (only after a direction is confirmed)

*The two branches below are sketched at a high level since the real shape
depends entirely on Phase 1's findings — treat these as starting guardrails,
not literal steps.*

### If own domain (`domains/medium/`):
- Follow the same shape as WO#33/#34: `models.py` importing `Base` from
  `core.base_model`; `routers/*.py` under 300 lines each; `templates/`
  extending `base.html`; `static/` only if needed.
- If extraction runs as a DAG: `airflow/dags/medium/`, `dag_db.py` only.
- Wire in: `main.py` router registration, `core/templating.py` loader path,
  static mount ordering, sidebar entry if in scope.

### If folded into `blog`:
- Work within `domains/blog/models.py` and `domains/blog/routers/blog.py`
  (or `blog_pipeline.py` if Track C's WO#27 split has already landed by the
  time this runs — check and use whichever exists) rather than creating
  parallel structures.
- If a new agent function is needed in `blog_agents.py`, follow the
  existing pattern (a system-instruction string, JSON schema where
  structured output is needed, called through `services.ai`) rather than
  the pre-consolidation approach the standalone Medium tool may have used.
- Decide whether Medium import is a new `BlogIdeaStatus`/`project_type`
  value, a new field on `BlogIdea`, or a genuinely separate flow that
  merely reuses the existing pipeline's agents — state which and why.

### Phase 2 Acceptance Criteria (adapt to whichever branch was taken)
- [ ] Implementation matches the direction confirmed at the end of Phase 1
  — if it diverged, the report explains why.
- [ ] Any AI/LLM calls route through `services/ai/`.
- [ ] If a DAG was added: confirmed it never imports `models.py`,
  `database.py`, or any router/service.
- [ ] Router file(s) under 300 lines.
- [ ] Design decisions documented.
- [ ] Basic functional smoke test: an end-to-end run (Medium input → final
  output) works.

## For the next work order (not part of this one)
None of Track E's other items depend on the direction chosen here.
