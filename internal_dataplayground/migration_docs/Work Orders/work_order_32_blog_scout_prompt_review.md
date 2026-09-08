# Work Order #32 — Blog Scout Prompt / Interest-Alignment Review

*Analysis task, not a code migration — the deliverable is a report. Any
resulting prompt change is explicitly separate, later, gated work. This is
the lowest-risk item in the entire backlog: Phase 1 (this work order) makes
zero code changes.*

*Revised from the original draft: input data arrives as a JSON export
(provided directly by the project owner) of a specific query against
`blog_ideas`, rather than a live `dag_db.py` connection. This removes any
need for DB credentials or read access as part of this work order.*

---

## ROLE
You are a data/product analyst reviewing an AI content-generation pipeline's
output against its stated inputs. You are not a prompt engineer in this work
order — you produce findings, not a rewritten prompt.

## HARD BOUNDARIES
- This work order's only required input is the JSON export described below
  and two files already in the repo (`blog_agents.py`,
  `life_os_blog_scout.py`). No database connection, no `dag_db.py` usage,
  no write access of any kind is needed or in scope.
- Do not propose or draft a revised prompt as part of this work order — if
  your analysis suggests one, describe *what* should change and *why* in
  the report, and stop there. A follow-up work order (not this one) would
  implement it, after review.
- If the export has too few analyzable rows for a conclusion to be
  meaningful, say so explicitly rather than drawing conclusions from a
  small sample.

## HANDLING PRE-EXISTING ISSUES
If something about the data looks like a bug elsewhere in the pipeline
(e.g. a status value that shouldn't exist, a null where one shouldn't be
possible): report it under Notes, don't fix it, don't let it block the
rest of the analysis.

## WORKING METHOD
Do the bucketing/categorization pass (Step 2) before comparing against
targets (Step 3) — get the raw distribution first, then check it against
what the prompt asks for.

## OUTPUT FORMAT
A short written report (see DELIVERABLE below) — not a code diff. No
"files created/moved/edited" section is expected for this work order.

## ROLLBACK
Not applicable — no files are changed.

---

## SCOPE

**Input data:** a JSON export provided directly by the project owner,
generated via:
```sql
SELECT id, title_concept, project_type, the_build, the_narrative,
       the_selling_point, raw_idea_input, status, difficulty
FROM blog_ideas
```
Treat this JSON as the full dataset — do not attempt to query the database
yourself.

**Note on what this query does *not* include:** no `created_at`/`updated_at`.
This means the analysis below is an aggregate, all-time distribution check,
not a recency/trend check ("did skew get worse over the last month"). That's
fine — do the aggregate analysis fully with what's provided, and just note
in the report that a recency/trend pass would need `created_at` added to a
future export, rather than treating its absence as a blocker.

**Read-only, already in the repo:**
- `airflow/agents/blog_agents.py` (`agent_researcher`'s system prompt and
  schema only)
- `airflow/dags/blog/life_os_blog_scout.py` (`DEFAULT_INTERESTS`,
  `DEFAULT_PROJECTS`, the difficulty-summary/dedup logic)

## Resolved before drafting (both were open questions in the master index)

- **Which agent function the Scout DAG calls:** `life_os_blog_scout.py`'s
  `task_run_researcher` calls `agent_researcher()` in `blog_agents.py`,
  passing `interests`, `existing_projects`, recent file narrations, existing
  idea titles (for dedup), and a `difficulty_context` string built from the
  last 14 days of generated ideas' difficulty mix.
- **Whether a canonical "likes and interests" baseline exists:** yes —
  `DEFAULT_INTERESTS` and `DEFAULT_PROJECTS` at the top of
  `life_os_blog_scout.py`. `DEFAULT_INTERESTS` is a single comma-separated
  string mixing sports leagues/teams (NBA, NFL, Premier League, MLB,
  Brasileirão, Champions League, Libertadores, Olympics), media (music, TV,
  movies, food), and technical topics (Tableau, SQL, AI, Gen AI, Python,
  Airflow, GCP, Gemini, BigQuery, data pipelines/extraction/analysis).
  `DEFAULT_PROJECTS` lists the six existing Life OS modules.

## One hypothesis to test, not assume

`DEFAULT_INTERESTS` names roughly **eleven** distinct sports/leagues in a
single comma list versus roughly **ten** technical/data topics, and
`agent_researcher`'s own system prompt requires "at least 2 of the 5 ideas
must be in domains OUTSIDE the author's existing projects" — sports is the
most obvious "outside" domain available given how the interest list is
weighted. If generated ideas have been skewing sports-heavy, the interest
list's own balance (not the prompt's balancing logic) may be the actual
cause. Test this against the real export rather than asserting it.

## STEPS

1. Parse the provided JSON export. `raw_idea_input` is populated only for
   BYOI (user-submitted) ideas — **exclude any row where `raw_idea_input` is
   not null/empty** from the Scout-specific analysis; this review is about
   what the Scout *generates*, not what the user submits.
2. Categorize each remaining title by rough topic domain (sports,
   music/media, food, technical/data, "existing Life OS module," other) —
   simple keyword bucketing is fine; state your bucketing rule so it's
   reproducible by someone else re-running this later.
3. Compare the resulting distribution against:
   - The stated 2-starter/2-weekend/1-ambitious per-batch difficulty target
     (using the `difficulty` column).
   - The stated 1-existing_asset/2-new_build/2-tutorial per-batch type
     target (using `project_type`).
   - The "at least 2 of 5 outside existing projects" rule (your domain
     bucketing from Step 2 against `DEFAULT_PROJECTS`' six modules).
4. Check for topic repetition — are the same 2–3 sports leagues or the same
   technical topics recurring disproportionately relative to how
   `DEFAULT_INTERESTS` weights them?
5. Check `status` — are there ideas stuck in `idea_generated` for what looks
   like an unusually long time relative to others (a proxy for "ideas that
   never got picked up," since there's no timestamp to measure directly)?
   Flag as a soft observation, not a hard finding, given the missing date
   field.
6. Spot-check for likely near-duplicate titles the dedup logic
   (`existing_lower` substring matching in `task_run_researcher`) may have
   missed — paraphrased repeats wouldn't be caught by a substring check.

## DELIVERABLE
A short report covering:
- Actual difficulty/type/domain distribution vs. stated targets, with real
  counts from the export.
- Whether `DEFAULT_INTERESTS`'s own topic balance (not the prompt's
  balancing instructions) appears to be driving any observed skew —
  explicitly confirm or rule out the sports-weighting hypothesis above with
  numbers.
- Any likely dedup misses found, with the specific title pairs.
- If a prompt or interest-list change looks warranted: describe the
  specific change and the evidence for it — do not implement it.
- One line noting that a recency/trend pass would need `created_at` added
  to a future export.

## ACCEPTANCE CRITERIA
- [ ] Every claim in the report is backed by an actual count from the
  export, not an impression.
- [ ] The sports-weighting hypothesis is explicitly tested and either
  confirmed or ruled out with numbers.
- [ ] BYOI rows (`raw_idea_input` populated) are excluded from the
  Scout-specific analysis, and the report states how many rows were
  excluded on that basis.
- [ ] No code or prompt changes made.
