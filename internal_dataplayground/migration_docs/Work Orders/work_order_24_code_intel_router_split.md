# Work Order #24 — Code Intel Domain Router Split (3 files: `ci_readme.py`, `ci_files.py`, `ci_projects.py`)

*Three independently-oversized files in one domain, per WO#19's lint script.
Treat as three sub-tasks (Parts A/B/C); verify each separately, but they can
land in one combined report since they share a domain and one cross-file
import needs updating across parts.*

---

## ROLE
You are a senior refactoring engineer splitting oversized router files by
responsibility. Location-and-organization refactor only — no behavior
change, no renamed endpoints, no schema change.

## HARD BOUNDARIES
- No endpoint's URL path, HTTP method, request/response shape, or template
  name changes.
- Only touch the files named in each Part's SCOPE, plus `main.py`.
- `main.py` currently registers `ci_files` and `ci_readme` **before**
  `ci_projects` with a comment explaining this is for path-matching
  specificity — preserve that relative ordering for whatever new files this
  split produces (state explicitly where you placed each new router and
  why).
- Confirm the real current line count of each file via the lint script
  before splitting it — don't split a file that's already resolved.

## HANDLING PRE-EXISTING BUGS
Don't fix, reproduce against baseline, report under Notes, mark ⚠️ not ❌.

## WORKING METHOD
Do Part A first (since Part C depends on its output), then B, then C.

## OUTPUT FORMAT
1. Files created
2. Files moved
3. Files edited (path — description)
4. Acceptance criteria results, grouped by Part
5. Notes

## ROLLBACK
`git checkout` on every file touched.

---

## Part A — `ci_readme.py`

### SCOPE
- `domains/code_intel/routers/ci_readme.py` (split)
- New: `domains/code_intel/routers/ci_folder_readme.py`
- `domains/code_intel/routers/ci_projects.py` (one import line updated —
  see Step 2)

### STEPS
1. Run the lint script; confirm the real current line count.
2. The folder-README lookup helpers (`_get_folder_readme`,
   `_get_latest_folder_readme` — already shared with `ci_projects.py`,
   which imports them) plus the two folder-README endpoints
   (`save_folder_readme`, `get_folder_readme`) are already a visually
   distinct block in the current file. Move all four into
   `ci_folder_readme.py`.
3. Update `ci_projects.py`'s existing `from
   domains.code_intel.routers.ci_readme import (_get_folder_readme,
   _get_latest_folder_readme)` to point at the new `ci_folder_readme`
   module — this is the one cross-file dependency to watch.
4. Keep `generate_readme`, `save_readme_edits`, `push_readme`,
   `trigger_readme_dag` in `ci_readme.py`.

---

## Part B — `ci_files.py`

### SCOPE
- `domains/code_intel/routers/ci_files.py` (split)
- New: `domains/code_intel/routers/ci_batch.py`

### STEPS
1. Run the lint script; confirm the real current line count.
2. Move the three batch-Airflow-trigger endpoints
   (`trigger_batch_narrate`, `trigger_batch_comment`,
   `trigger_batch_improve`) into `ci_batch.py`.
3. Keep the inline single-file endpoints (`file_detail`, `pull_file`,
   `narrate_file`, `comment_file`, `improve_file`, `push_commented_file`,
   `update_comment_status`) in `ci_files.py` — these carry the heaviest
   docstrings, which is most of why this file is long; keep the
   docstrings, they're doing real documentation work per this codebase's
   conventions.

---

## Part C — `ci_projects.py`

### SCOPE
- `domains/code_intel/routers/ci_projects.py` (light, conditional split)
- New (only if needed): `domains/code_intel/routers/ci_status.py`

### STEPS
1. Run the lint script **after Parts A and B have landed** — this file is
   the most borderline of the three, and freeing up shared imports from
   the other two parts may already resolve it.
2. If still over 300: move the two polling/status endpoints
   (`get_file_statuses`, `project_status`) into `ci_status.py` — a
   cohesive "polling API for the frontend's badge-refresh loop" concern,
   distinct from project CRUD (`project_list_ui`, `create_project`,
   `delete_project`, `sync_files_from_github`, `project_detail`).
3. If already under 300: skip, report as not needed, and say so explicitly.

---

## ACCEPTANCE CRITERIA (all three parts)
- [ ] All resulting files under 300 lines (or Part C explicitly confirmed
  as not needed, with the real line count reported).
- [ ] `ci_projects.py`'s import of the folder-readme helpers correctly
  repointed to `ci_folder_readme.py`.
- [ ] Every Code Intel endpoint (project CRUD, sync, README
  generate/save/push/trigger, folder-README save/get, per-file
  pull/narrate/comment/improve/push, batch triggers, status/polling) still
  reachable at its original path with identical behavior.
- [ ] `main.py`'s `ci_files`/`ci_readme`-before-`ci_projects` ordering
  constraint (for path-matching specificity) confirmed to still hold with
  the new files added; state where `ci_folder_readme`, `ci_batch`, and
  `ci_status` were registered and why.

## For the next work order (not part of this one)
Can run fully in parallel with every other Track C work order, Track A,
Track B, and Track D.
