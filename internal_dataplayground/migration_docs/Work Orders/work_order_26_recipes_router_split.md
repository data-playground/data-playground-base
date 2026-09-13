# Work Order #26 — Recipes Domain Router Split (2 files: `recipe_extract.py`, `recipes.py`)

---

## ROLE
You are a senior refactoring engineer splitting oversized router files by
responsibility. Location-and-organization refactor only — no behavior
change, no schema change.

## HARD BOUNDARIES
- No endpoint's URL path, HTTP method, request/response shape, or template
  name changes.
- Only touch the files named in each Part's SCOPE, plus `main.py` if route
  registration ordering needs revisiting (it shouldn't — see Part A/B's own
  notes).
- `main.py` currently registers `recipe_extract` and `recipe_discovery`
  before `pantry`, which registers before `recipes` (which owns a catch-all
  `/{id}` route) — preserve this relative ordering.
- Do not fix, only report: `recipes.py`'s `update_recipe()` currently uses
  `__import__("sqlalchemy", fromlist=["delete"]).delete(...)` inline
  instead of a top-level `from sqlalchemy import delete`. This is a
  pre-existing style oddity, unrelated to line count — flag it under Notes
  for a separate trivial ticket, do not fix it here.

## HANDLING PRE-EXISTING BUGS
Don't fix, reproduce against baseline, report under Notes, mark ⚠️ not ❌.
(The `__import__` oddity above is a specific, pre-identified instance of
this rule.)

## WORKING METHOD
Run `scripts/check_router_line_limits.py` first for both files; confirm
real current line counts before proceeding with either part.

## OUTPUT FORMAT
1. Files created
2. Files moved
3. Files edited (path — description)
4. Acceptance criteria results, grouped by Part
5. Notes

## ROLLBACK
`git checkout` on every file touched.

---

## Part A — `recipe_extract.py`

### SCOPE
- `domains/recipes/routers/recipe_extract.py` (split)
- New: `domains/recipes/routers/_extraction_helpers.py`

### STEPS
1. Move `_fetch_url_content`, `_parse_schema_org` (the largest single
   function in this file — handles JSON-LD parsing across many recipe-site
   schema variants), and `_strip_html` into `_extraction_helpers.py`. None
   of these have route decorators; they're pure extraction utilities.
2. Keep the four route handlers (`extract_landing`, `extract_from_url`,
   `extract_from_file`, `confirm_extraction`) in `recipe_extract.py`.
3. Preserve the module's TODO/Playwright integration-point comments — move
   them to wherever `_fetch_url_content` ends up (`_extraction_helpers.py`),
   since that's the function the comment is actually about.

---

## Part B — `recipes.py`

### SCOPE
- `domains/recipes/routers/recipes.py` (split)
- New: `domains/recipes/routers/recipe_mutations.py`

### STEPS
1. Move the three small, single-purpose mutation endpoints (`rate_recipe`,
   `toggle_favorite`, `log_cook` — each returns a tiny partial and does one
   thing) into `recipe_mutations.py`.
2. Keep `recipe_library`, `list_tags`, `suggest_ingredients`,
   `recipe_detail`, `create_recipe`, `update_recipe`, `delete_recipe` in
   `recipes.py`.
3. See HARD BOUNDARIES regarding `update_recipe()`'s inline `__import__` —
   report, don't fix.

---

## ACCEPTANCE CRITERIA (both parts)
- [ ] All resulting files under 300 lines.
- [ ] `GET /recipes`, `GET /recipes/tags`, `GET
  /recipes/ingredients/suggest`, `GET /recipes/{id}`, `POST /recipes`,
  `PATCH /recipes/{id}` (including the `replace_ingredients` path), `DELETE
  /recipes/{id}` unchanged.
- [ ] `PATCH /recipes/{id}/rate`, `PATCH /recipes/{id}/favorite`, `POST
  /recipes/{id}/cook` unchanged, still return their respective
  micro-partials.
- [ ] `GET /recipes/extract`, `POST /recipes/extract/url`, `POST
  /recipes/extract/file`, `POST /recipes/extract/confirm` unchanged
  (Schema.org and Gemini-fallback extraction paths mocked/stubbed per the
  usual external-API caveat — mark ⚠️ if live verification isn't
  available).
- [ ] `main.py`'s registration ordering (`recipe_extract`/
  `recipe_discovery` → `pantry` → `recipes`) confirmed to still hold with
  the new files added.

## For the next work order (not part of this one)
Can run fully in parallel with every other Track C work order, Track A,
Track B, and Track D.
