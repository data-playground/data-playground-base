# Work Order #7 — Domain Migration: `recipes` (+ `pantry`)

*Moved together — `pantry` is not a separate conceptual domain, it's a
thin view over `Ingredient`/`PantryItem`, which live in the same model
group as `Recipe`. This work order has more relationship complexity than
any prior one: a many-to-many association table (`recipe_tags_junction`)
that is a plain `sqlalchemy.Table`, not a mapped class, plus a real
cross-domain relationship reaching into the not-yet-migrated `planning`
domain (`WeeklyPlanMeal.recipe_id` → `Recipe`). Read this whole work order
before starting — the association-table handling in Step 3 is the part
most likely to go wrong if rushed.*

---

## ROLE
You are a senior refactoring engineer performing a structural code migration.
Your job is NOT to improve, optimize, or modernize the code you move — only
to relocate it correctly and verify it still behaves identically. Resist the
urge to "clean up while you're in there." Flag improvement opportunities as
a NOTES section at the end instead of acting on them.

## HARD BOUNDARIES
- Only read/edit files explicitly listed in SCOPE below. If you believe you
  need to touch a file outside this list to complete the task, STOP and
  report why — do not proceed and do not guess.
- No schema changes. No renamed tables, columns, or endpoints. No behavior
  changes. This is a location refactor only.
- If a step's instructions conflict with what you find in the actual code
  (e.g. a file/class isn't where the work order says it is), stop and report
  the discrepancy rather than improvising a fix.
- You may create empty `__init__.py` package markers as needed to support
  new import paths — this is expected scaffolding, not a scope expansion,
  and does not need to be flagged as a deviation (a one-line mention in the
  report's "Files created" section is enough).
- **`airflow/agents/recipe_agents.py` is explicitly OUT OF SCOPE and must
  NOT be moved.** It contains the Gemini-based extraction/normalization/
  discovery agent functions and is imported directly by
  `recipe_extract.py`, `recipe_discovery.py`, and `services/recipe_service.py`
  — leave all of those import statements pointing at
  `airflow.agents.recipe_agents` exactly as they are today. This file
  contains no `models` imports, so it needs no internal edits either.
- **`services/recipe_service.py` stays at its current path — do not move
  it** (it's shared infrastructure, same tier as `github_service.py`). It
  DOES need an internal import-path edit, though (see Step 5), because
  unlike `github_service.py` it directly imports domain model classes. This
  is the one file in this work order that is "edited but not moved."
- **`recipe_tags_junction` is a `sqlalchemy.Table` object, not an ORM
  class.** It must move to `domains/recipes/models.py` alongside `Recipe`
  and `RecipeTag`, but do not attempt to treat it like a class — it has no
  `Base` subclass, it's built directly from `Base.metadata` via
  `sqlalchemy.Table(...)`. Move the `Column`/`Table` import additions and
  the table definition exactly as they appear, in the same relative
  position (after `RecipeTag`, before `Recipe`, per the current file
  layout) — do not reorder it, since `Recipe.tags` and `RecipeTag.recipes`
  both reference it by variable name at class-definition time, so it must
  be defined before both.
- `templates/partials/recipe_micro_partials.html` appears to be a stale
  duplicate — it concatenates the full content of `recipe_rating.html`,
  `recipe_favorite.html`, and `recipe_cook_count.html` (which also exist as
  separate, individually-referenced files) into one file under one name
  that doesn't match anything any router actually requests via
  `TemplateResponse(...)`. **Move it anyway** as part of this domain (don't
  leave it behind), but do not delete it and do not assume it's dead
  without confirming — search the codebase for any reference to
  `"partials/recipe_micro_partials.html"` and report what you find, the
  same way WO#5 required confirming `account_options.html`'s usage.
- This domain has **no dedicated static CSS file** — all recipe/pantry
  page styling is inline (`extra_css` blocks within each template). Do not
  create one; there is nothing to move in the static asset category for
  this domain.

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
If, while verifying an acceptance criterion, you discover the app behaves
incorrectly in a way that is unrelated to this migration:
1. Do NOT fix it — it is out of scope by default even if the fix looks trivial.
2. Reproduce the same failure against the pre-migration baseline (the code
   as it existed before your changes) to confirm it is not a regression you
   introduced.
3. Report it under "Notes" with enough detail to file a standalone ticket:
   which endpoint/file, the exact error, and root cause if you found one.
4. Mark the related acceptance criterion ⚠️ (not ❌) if the criterion's
   *intent* is otherwise satisfied but blocked by this pre-existing issue —
   explain the distinction in the one-line reason.

## WORKING METHOD
Execute steps in the order listed. After each step that changes running
behavior (not pure file moves), pause and self-verify against the relevant
acceptance criteria before continuing to the next step. Do not defer all
verification to the end.

If an acceptance criterion requires a resource not listed in SCOPE (e.g. a
config file, external service, or database not provided), do not skip the
criterion silently. Perform the closest verification achievable with what
you have, state explicitly what the substitute check was and why, and mark
the result ⚠️ rather than ✅ or leaving it blank. (Note: `recipe_extract.py`'s
URL/PDF/image extraction and `recipe_discovery.py`'s Gemini-based
suggestions cannot be verified against live external calls in this
environment — verify routes are reachable and correctly call into
`airflow.agents.recipe_agents` with a mocked/stubbed response, same pattern
as the GitHub/ATS/Gemini caveats in prior work orders.)

## OUTPUT FORMAT
End with a report in exactly this structure:
1. **Files created** (list — include any scaffolding `__init__.py` files here)
2. **Files moved** (old path → new path)
3. **Files edited** (path — one-line description of the change; if a change
   goes beyond what a step literally asked for but was necessary to make
   that step work, say so explicitly and explain why)
4. **Acceptance criteria results** (checklist, ✅/❌/⚠️ per item, with a
   one-line reason for any ❌ or ⚠️ — including substitute-verification
   explanations per the rule above)
5. **Notes / things I noticed but did not act on** (optional — improvement
   ideas, pre-existing bugs found per the rule above, inconsistencies, risks
   spotted while working, explicitly out of scope for this task)

## ROLLBACK
This work order operates on files already tracked in git. If acceptance
criteria fail and cannot be quickly fixed, the safe rollback is `git
checkout` on every file listed in "Files created / moved / edited" above —
do not attempt a partial manual revert.

---

## SCOPE

**Models to extract from `models.py`:**
`IngredientCategory`, `RecipeSourceType`, `RecipeMealType`,
`RecipeDifficulty`, `IngredientUnit`, `Ingredient`, `RecipeTag`,
`recipe_tags_junction` (Table object — see HARD BOUNDARIES), `Recipe`,
`RecipeIngredient`, `PantryItem`, `IngredientResponse`,
`RecipeIngredientResponse`, `RecipeTagResponse`, `RecipeResponse`,
`RecipeCreate`, `PantryItemResponse`

**Routers:**
- `routers/recipe_extract.py`
- `routers/recipe_discovery.py`
- `routers/pantry.py`
- `routers/recipes.py`

**Templates:**
- `templates/recipes.html`
- `templates/recipe_extract.html`
- `templates/recipe_discover.html`
- `templates/recipe_detail.html`
- `templates/pantry.html`
- `templates/partials/recipe_extract_preview.html`
- `templates/partials/discovery_results.html`
- `templates/partials/discovery_save_result.html`
- `templates/partials/recipe_rating.html`
- `templates/partials/recipe_favorite.html`
- `templates/partials/recipe_cook_count.html`
- `templates/partials/pantry_list.html`
- `templates/partials/recipe_micro_partials.html` (see HARD BOUNDARIES —
  move and report on usage, do not delete)

**Static:** none — see HARD BOUNDARIES.

**Core/config files to edit:**
- `models.py`
- `main.py`
- `core/templating.py`
- `services/recipe_service.py` (edited in place — internal import path
  only, file does NOT move; see Step 5)

**Not in scope, referenced only to confirm no breakage:**
- `airflow/agents/recipe_agents.py` (imported by three files in this
  domain plus `recipe_service.py` — must remain importable at its current
  path, unchanged)
- `routers/weekly_plan.py` (planning domain, not yet migrated — imports
  `Recipe`, `RecipeMealType`, `Ingredient`, `PantryItem` from `models` at
  module level, plus a local `from models import RecipeIngredient` inside
  `_generate_shopping_list()`. `WeeklyPlanMeal.recipe_id` also has a real
  `relationship("Recipe", ...)`, not just a bare FK column — this is the
  one genuine cross-domain relationship in this work order. All of these
  must keep resolving via the shim; see Step 6.)

---

## STEPS

1. **Create `domains/recipes/models.py`.** Move the five enums
   (`IngredientCategory`, `RecipeSourceType`, `RecipeMealType`,
   `RecipeDifficulty`, `IngredientUnit`) first, then `Ingredient`, then
   `RecipeTag`, then `recipe_tags_junction` (the `Table` object — including
   its `from sqlalchemy import Column, Table` import line, which currently
   sits right above the table definition in the source file), then
   `Recipe`, `RecipeIngredient`, `PantryItem`, then the six Pydantic
   schemas — preserve this exact relative order, since `Recipe.tags` and
   `RecipeTag.recipes` both reference `recipe_tags_junction` by name and it
   must be defined before either class body executes. Import `Base` from
   `core.base_model`.

2. **Confirm same-file relationship strings resolve.** `Recipe.ingredients`
   ↔ `RecipeIngredient.recipe`, `RecipeIngredient.ingredient` ↔
   `Ingredient.recipe_ingredients`, `Ingredient.pantry_item` ↔
   `PantryItem.ingredient`, and `Recipe.tags` ↔ `RecipeTag.recipes` (via
   `secondary=recipe_tags_junction`) are all same-module references once
   this file is assembled — no special cross-file handling needed, same as
   the `Job`/`ApplicationLog` case in WO#3. Just confirm nothing was left
   behind in root `models.py` that any of these five classes still depend
   on.

3. **In `models.py`:** delete the seventeen moved definitions (5 enums + 6
   classes/table + 6 Pydantic schemas — recount against your actual move to
   make sure nothing is missed) and replace with a re-export shim:
   `from domains.recipes.models import IngredientCategory,
   RecipeSourceType, RecipeMealType, RecipeDifficulty, IngredientUnit,
   Ingredient, RecipeTag, recipe_tags_junction, Recipe, RecipeIngredient,
   PantryItem, IngredientResponse, RecipeIngredientResponse,
   RecipeTagResponse, RecipeResponse, RecipeCreate, PantryItemResponse`.
   Tag it `# TODO: remove after all cross-references are updated`.

4. **Move routers:**
   - `routers/recipe_extract.py` → `domains/recipes/routers/recipe_extract.py`
   - `routers/recipe_discovery.py` → `domains/recipes/routers/recipe_discovery.py`
   - `routers/pantry.py` → `domains/recipes/routers/pantry.py`
   - `routers/recipes.py` → `domains/recipes/routers/recipes.py`

   Update each file's model imports to pull from `domains.recipes.models`
   instead of `models`, **including the local, in-function import inside
   `recipes.py`'s `update_recipe()`** (`from models import
   RecipeIngredient` → `from domains.recipes.models import
   RecipeIngredient`). Update each file's `templates =
   Jinja2Templates(directory="templates")` to `from core.templating import
   templates`. Leave every `from airflow.agents.recipe_agents import ...`
   and `from services.recipe_service import run_normalization_pipeline`
   import unchanged.

5. **Edit `services/recipe_service.py` in place (do not move it).** Its
   `run_normalization_pipeline()` function has a local, in-function import:
   `from models import Ingredient, IngredientCategory, RecipeIngredient,
   IngredientUnit, RecipeTag`. Update this to `from domains.recipes.models
   import Ingredient, IngredientCategory, RecipeIngredient, IngredientUnit,
   RecipeTag`. This is the only change to this file — its physical location
   under `services/` does not change.

6. **Move templates**, preserving the `partials/` subfolder structure, into
   `domains/recipes/templates/` per the SCOPE list above.

7. **Update `core/templating.py`'s `ChoiceLoader`** to add
   `domains/recipes/templates/` as an additional search root, alongside the
   roots already added in WO#1–6.

8. **In `main.py`:**
   - Update the four router imports/includes to their new paths (`from
     domains.recipes.routers import recipe_extract, recipe_discovery,
     pantry, recipes`).
   - **Preserve the existing include order and the comment explaining
     it** — `main.py` currently registers these with a comment noting
     `recipe_extract` and `recipe_discovery` must come before `pantry`,
     which must come before `recipes` (which has a catch-all `/{id}`
     route). Keep that exact relative ordering after the import path
     change.
   - This domain has no static assets (see HARD BOUNDARIES), so **no new
     `StaticFiles` mount is needed** — do not add one.

---

## ACCEPTANCE CRITERIA

- [ ] `GET /recipes` renders identically — filter panel, masonry grid,
  empty state
- [ ] `GET /recipes/extract` renders identically — all three tabs (URL,
  PDF/Photo, Manual)
- [ ] `GET /recipes/discover` renders identically — both pantry-mode and
  open-mode panes, pantry-empty warning logic intact
- [ ] `GET /pantry` renders identically — add form, autocomplete dropdown
  markup, category-grouped ingredient list
- [ ] `GET /recipes/{id}` renders identically for an existing recipe —
  servings scaler, ingredient list, instructions (both normal and step
  modes), edit panel
- [ ] `POST /recipes` (manual creation) still redirects to the new recipe's
  detail page and the normalization pipeline call still reaches
  `services.recipe_service.run_normalization_pipeline` correctly (DB writes
  can be verified directly; the Gemini normalization call itself may be
  mocked/stubbed per the WORKING METHOD note)
- [ ] `PATCH /recipes/{id}` (including the `replace_ingredients` path,
  which calls the local `RecipeIngredient` import inside `update_recipe()`)
  still works correctly — this specifically exercises the import fixed in
  Step 4
- [ ] `PATCH /recipes/{id}/rate`, `/favorite`, and `POST /recipes/{id}/cook`
  still return their respective micro-partials (`recipe_rating.html`,
  `recipe_favorite.html`, `recipe_cook_count.html`) correctly
- [ ] `POST /pantry`, `DELETE /pantry/{ingredient_id}`, and `GET
  /pantry/suggest` all still work and return `pantry_list.html` /
  autocomplete JSON correctly
- [ ] `POST /recipes/discover/pantry` and `/recipes/discover/open` reach
  their respective agent calls correctly (mocked/stubbed per WORKING
  METHOD); `POST /recipes/discover/save` still saves and returns
  `discovery_save_result.html` correctly
- [ ] `Base.metadata` table-identity check (method established in WO#1):
  same table count before/after, `models.Recipe is
  domains.recipes.models.Recipe`, `models.recipe_tags_junction is
  domains.recipes.models.recipe_tags_junction` (confirm this specifically
  for the `Table` object, not just the ORM classes), no
  `InvalidRequestError` on mapper configuration
- [ ] `routers/weekly_plan.py`'s existing `Recipe` / `RecipeMealType` /
  `Ingredient` / `PantryItem` module-level imports and its local
  `RecipeIngredient` import inside `_generate_shopping_list()` all still
  resolve via the shim; if a weekly plan with linked recipes exists or can
  be created in this environment, confirm shopping-list generation
  (`_generate_shopping_list`) still correctly resolves `Recipe` ↔
  `WeeklyPlanMeal` — otherwise mark ⚠️ and explain what couldn't be tested
- [ ] `grep -r "from models import"` for each of the moved names across the
  repo returns only the shim's own lines in `models.py`, the two
  `weekly_plan.py` references noted above, and `services/recipe_service.py`
  (now pointing at `domains.recipes.models` per Step 5, not `models`) —
  nothing else should need updating
- [ ] Confirm and report whether `templates/partials/recipe_micro_partials.html`
  is referenced anywhere (see HARD BOUNDARIES) — required part of this report

---

## For the next work order (not part of this one)

Per GOVERNANCE.md §3.3, **Work Order #8 = `workout`** is next (routers:
`workout`, `workout_log`, `workout_plans`, `workout_settings` — note
`workout_log.py` registers *two* routers from one file,
`router` and `body_metrics_router`, both included separately in `main.py`
today — carry that forward exactly). After that, **Work Order #9 = `media`**,
then **Work Order #10 = `planning`** (`weekly_plan.py` + `intent.py`) last
among the real domains, specifically *because* it's the one with the most
outstanding cross-domain references into domains already migrated in
WO#5–9 (recipes, workout) — by the time it's tackled, those shims will
already exist and this work order's own local-import notes (the
`RecipeIngredient` and `Recipe`/`Ingredient`/`PantryItem` references) will
finally get resolved to their true `domains.recipes.models` location as
part of *that* migration, not this one. `weekly_plan.py` is also already
past the 300-line rule (GOVERNANCE.md §1.2) and will likely need splitting
as part of that move, not just relocation — flag that explicitly when
drafting Work Order #10.
