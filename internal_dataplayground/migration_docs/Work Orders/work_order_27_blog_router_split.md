# Work Order #27 — Blog Domain Router Split (`domains/blog/routers/blog.py`)

---

## ROLE
You are a senior refactoring engineer splitting an oversized router file by
responsibility. Location-and-organization refactor only — no behavior
change, no schema change.

## HARD BOUNDARIES
- No endpoint's URL path, HTTP method, request/response shape, or template
  name changes.
- Only touch `domains/blog/routers/blog.py`, the new file you create, and
  `main.py` for router registration.
- `airflow.agents.blog_agents` is not imported by this domain's routers at
  all today — confirm this stays true; this split does not add any new
  import of that module, only reorganizes which router function calls
  `trigger_airflow(...)`.

## HANDLING PRE-EXISTING BUGS
Don't fix, reproduce against baseline, report under Notes, mark ⚠️ not ❌.

## WORKING METHOD
Run `scripts/check_router_line_limits.py` first; confirm the real current
line count before proceeding.

## OUTPUT FORMAT
1. Files created
2. Files moved
3. Files edited (path — description)
4. Acceptance criteria results
5. Notes

## ROLLBACK
`git checkout` on every file touched.

---

## SCOPE
- `domains/blog/routers/blog.py` (split)
- New: `domains/blog/routers/blog_pipeline.py`
- `main.py`

## STEPS

1. Run the lint script; confirm the real current line count.

2. Move the HITL/Airflow-pipeline endpoints — `save_evidence`,
   `trigger_creator`, `save_review`, `trigger_finalizer` — into
   `blog_pipeline.py`. These are the four endpoints that call
   `trigger_airflow(...)` and represent the Ghostwriter/Refiner/Editor
   pipeline steps specifically.

3. Keep the kanban/CRUD surface (`blog_kanban`, `create_idea`,
   `idea_detail`, `update_status`, `archive_idea`, `delete_idea`,
   `revert_idea_status`, `trigger_scout`, `view_article`) in `blog.py`.

4. Both files share the `/blog` prefix — confirm no path collisions (list
   every route from both files in your report).

5. Register `blog_pipeline`'s router in `main.py` alongside the existing
   `blog.router` include.

## ACCEPTANCE CRITERIA
- [ ] Both files under 300 lines.
- [ ] `GET /blog`, `POST /blog/ideas`, `GET /blog/ideas/{id}`, `PATCH
  /blog/ideas/{id}/status`, `PATCH /blog/ideas/{id}/archive`, `DELETE
  /blog/ideas/{id}`, `PATCH /blog/ideas/{id}/revert`, `POST /blog/scout`,
  `GET /blog/ideas/{id}/article` unchanged.
- [ ] `PATCH /blog/ideas/{id}/evidence`, `PATCH /blog/ideas/{id}/trigger`,
  `PATCH /blog/ideas/{id}/review`, `PATCH /blog/ideas/{id}/finalize`
  unchanged — each still triggers its correct DAG
  (`life_os_blog_creator`/`life_os_blog_finalizer`) with the same conf.
- [ ] `airflow.agents.blog_agents` confirmed still not imported by either
  resulting router file.

## For the next work order (not part of this one)
Can run fully in parallel with every other Track C work order, Track A,
Track B, and Track D.
