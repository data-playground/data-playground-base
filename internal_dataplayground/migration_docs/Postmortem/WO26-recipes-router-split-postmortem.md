# Work Order #26 — Recipes Domain Router Split: Post-Mortem

**Status:** Part A and Part B delivered against the written work order. A
second, separately-agreed changeset (six fixes surfaced during review) was
then applied to the same files. Both are documented here because they must
be evaluated — and, if committed, landed — differently. See §4's preface
before treating this as one diff.

**Purpose of this document:** so a reviewer (human or AI) can tell exactly
what belongs to WO#26 as written, what was added afterward and why, and
what is still open, without re-reading the full review conversation. This
document is written to be consumed directly by a future coding agent
picking up the follow-on work in §8 — it does not assume that agent has
this conversation's context.

---

## 1. What WO#26 Actually Asked For

Recap of the written work order (`work_order_26_recipes_router_split.md`):

- **Role:** refactoring engineer performing a location-and-organization
  split only — no behavior change, no schema change.
- **Part A:** split `recipe_extract.py` by moving the three
  route-decorator-free helper functions (`_fetch_url_content`,
  `_parse_schema_org`, `_strip_html`) into a new
  `_extraction_helpers.py`, keeping the four route handlers in
  `recipe_extract.py`. The Playwright TODO comment moves with
  `_fetch_url_content`, since that's the function it actually describes.
- **Part B:** split `recipes.py` by moving the three small mutation
  endpoints (`rate_recipe`, `toggle_favorite`, `log_cook`) into a new
  `recipe_mutations.py`, keeping `recipe_library`, `list_tags`,
  `suggest_ingredients`, `recipe_detail`, `create_recipe`,
  `update_recipe`, `delete_recipe` in `recipes.py`.
- **Explicit non-goal:** the WO pre-identifies `update_recipe()`'s inline
  `__import__("sqlalchemy", fromlist=["delete"]).delete(...)` as a known
  style oddity and instructs the agent **not** to fix it, only report it —
  it was "unrelated to line count."
- **Hard boundary:** only the files named in each Part's SCOPE, plus
  `main.py` if registration ordering needed revisiting (the WO's own text
  says it shouldn't).
- **Acceptance criteria:** all resulting files under 300 lines; every
  listed endpoint's path/method/response shape unchanged; `main.py`'s
  registration ordering (`recipe_extract`/`recipe_discovery` → `pantry` →
  `recipes`) still holds with the new files added.

## 2. What Was Delivered Under the Original Work Order

### Part A
- **Created:** `domains/recipes/routers/_extraction_helpers.py` (203
  lines) — `_fetch_url_content`, `_parse_schema_org`, `_strip_html`,
  copied verbatim, with the Playwright TODO block moved here.
- **Edited:** `domains/recipes/routers/recipe_extract.py` — the four
  route handlers stayed, now importing the three helpers from the new
  module.

### Part B
- **Created:** `domains/recipes/routers/recipe_mutations.py` (107 lines
  after the later hardening pass; 104 as originally split) —
  `rate_recipe`, `toggle_favorite`, `log_cook`, copied verbatim, keeping
  the router's `prefix="/recipes"` and `tags=["Recipes"]` so OpenAPI
  grouping is unchanged.
- **Edited:** `domains/recipes/routers/recipes.py` — the three mutation
  endpoints removed; everything else in the WO's "keep" list stayed.

### `main.py` — deviation from the letter of "don't touch this file"
The WO says `main.py` shouldn't need edits. In practice, registering
`recipe_mutations.router` is required or `rate`/`favorite`/`cook` all
404 — this isn't optional. The suggested diff:

```diff
-from domains.recipes.routers import recipe_extract, recipe_discovery, pantry, recipes # WO7
+from domains.recipes.routers import recipe_extract, recipe_discovery, pantry, recipes, recipe_mutations # WO7, WO#26
 ...
 app.include_router(recipes.router)            # ← NEW — last, has /{id} catch-all
+app.include_router(recipe_mutations.router)   # WO#26 — /{id}/rate|favorite|cook
```

This was placed after `recipes.router` specifically because the WO's
acceptance criteria require the existing relative ordering
(`recipe_extract`/`recipe_discovery` → `pantry` → `recipes`) to still
hold, and `/rate`, `/favorite`, `/cook` have a different path shape than
`/{recipe_id}` so they can't be shadowed by it either way. **This diff
has not been applied to a real `main.py`** — no repository was available
in this session (see §3). Apply it, then run the app once, before
treating Part B as verified end-to-end.

### Acceptance criteria as originally reported

| # | Criterion | Result | Note |
|---|---|---|---|
| 1 | All resulting files under 300 lines | ✅ | 203 / 266 (280 after §4) / 104 (107 after §4) / 293 (292 after §4) |
| 2 | `scripts/check_router_line_limits.py` run first, real baselines confirmed | ⚠️ | Script wasn't provided; measured with `wc -l` instead. Pre-split baselines for the two original files were never independently confirmed. |
| 3 | Read-endpoint set unchanged (`GET /recipes`, `/tags`, `/ingredients/suggest`, `/{id}`, `POST /recipes`, `PATCH /{id}` incl. `replace_ingredients`, `DELETE /{id}`) | ✅ | Verified via OpenAPI route table against a stub app; bodies copied verbatim. |
| 4 | Mutation endpoints unchanged, still return their micro-partials | ✅ | Verified against a fake DB session; correct partial names returned for rate/favorite/cook. |
| 5 | Extraction endpoints unchanged | ✅ / ⚠️ | Schema.org path, Gemini-fallback path, JS-shell-too-short path, and fetch-failure path all verified against stubs. PDF and image branches verified only after §4's changes; live extraction (real HTTP, real Gemini) was never exercised — flagged per the WO's own external-API caveat. |
| 6 | `main.py` ordering holds | ⚠️ | Confirmed in a reduced stand-in app, not the real `main.py`. The diff above needs to actually be applied and the real app run once. |

## 3. Verification Method and Its Limits

No repository was available in this session — `/mnt/user-data/uploads`
was empty, and everything came from documents pasted directly into the
conversation. To verify the split without a real repo, a stand-in FastAPI
app was built with stub replacements for `database.py`,
`core/templating.py`, `services/recipe_service.py`,
`airflow/agents/recipe_agents.py`, and `domains/recipes/models.py`, and
the following was run against it:

- `pyflakes` across all four touched files (clean, exit 0, after §4).
- An OpenAPI route-table diff against the WO's acceptance-criteria list —
  exact match, 14 routes.
- Scenario tests: Schema.org happy path (including `@graph`, duration
  parsing, yield parsing, image parsing), Gemini fallback, the
  JS-shell-too-short error, the fetch-failure error, the unsupported-file
  error, and (after §4) agent-raises and agent-returns-`None` for all
  three extraction call sites, plus `rate_recipe`'s non-numeric/empty/
  out-of-range/valid inputs, plus `update_recipe`'s `replace_ingredients`
  path against a properly SQLAlchemy-mapped stand-in model.

**This is not integration testing.** It does not exercise the real
database, the real Gemini/agent implementations, or real Jinja2 template
rendering — every `TemplateResponse` call was intercepted by a stub that
returns JSON instead of rendering `partials/recipe_extract_preview.html`,
`recipe_rating.html`, etc. The route bodies are verbatim copies of the
originals, which bounds the risk, but a real run of the app against a
real database and the real templates is still the correct final gate
before merging.

## 4. Post-Migration Changes Agreed During Review

**These nine items are not part of WO#26's original scope, and were not
in its acceptance criteria.** They surfaced while reviewing WO#26's
output, in a conversation turn after the migration report was delivered.
Per this project's own process rule (GOVERNANCE.md §4.5, "Bugs Found
During Migration Are Not Migration Work"), work discovered this way is
supposed to get its own standalone ticket and its own diff — "never
bundled into the migration's diff, even if the fix is one line" — so
that a migration's diff stays reviewable as pure relocation and a bug fix
stays independently revertible. For expediency within this session, six
of the nine fixes below were applied directly into the same output files
as the Part A/B split, at the requester's explicit direction. **The
reviewer should not read that as this changeset satisfying WO#26's own
acceptance criteria** — it's a second, separate changeset that happens to
share a diff with the first one in this session's output. See §7 for how
to actually separate them before merging.

| # | Flagged | Decision | Change | File(s) | How verified |
|---|---|---|---|---|---|
| 1 | `recipes.py` at ~293 lines, past GOVERNANCE §1.2's 250-line "split before crossing 300" guidance | Leave as-is | None | — | — (tracked forward in §8.4) |
| 2 | `update_recipe()`'s `__import__("sqlalchemy", fromlist=["delete"]).delete(...)`, explicitly called out in the WO as pre-existing and out of scope | Fix | Replaced with a top-level `from sqlalchemy import delete` and `delete(RecipeIngredient).where(...)` | `recipes.py` | Exercised the `replace_ingredients` branch against a stub DB with a real SQLAlchemy-mapped `RecipeIngredient` stand-in; confirmed it builds and would execute a real `Delete` construct (a non-mapped stub correctly raised `ArgumentError`, proving the check is meaningful, not just "it imports") |
| 3 | New files (`_extraction_helpers.py`, `recipe_mutations.py`) don't roll back via a plain `git checkout` | Acknowledge | None | — | Documented correct rollback in §7 |
| 4 | Unused imports: `HTTPException` (`recipe_extract.py`), `or_` and `IngredientCategory` (`recipes.py`) — all three pre-existing in the originals, not introduced by the split | Fix | Removed all three | `recipe_extract.py`, `recipes.py` | `pyflakes` clean (exit 0) on all four files |
| 5 | `def parse_duration(s: any)` — the builtin `any()` used where `typing.Any` was meant | Fix | Added `from typing import Any`, changed the annotation | `_extraction_helpers.py` | No runtime behavior change (annotations aren't enforced at runtime); this only matters for a future static-typing pass |
| 6 | Stale "requests.get()" wording in two docstrings, though the implementation uses `httpx.AsyncClient` | Fix | Corrected both occurrences (module docstring in `recipe_extract.py`; Playwright TODO block in `_extraction_helpers.py`) | `recipe_extract.py`, `_extraction_helpers.py` | Comment-only change, no behavior to test |
| 7 | `rate_recipe`'s `int(form.get("rating", 0))` raises `ValueError` → unhandled 500 on non-numeric input, instead of the intended 422 | Fix | Wrapped in `try/except (TypeError, ValueError)` returning the same 422 | `recipe_mutations.py` | `rating="abc"` and missing `rating` now both return 422; `rating="9"` still 422; `rating="4"` still succeeds and returns the rating partial |
| 8 | The three extraction-agent calls (`agent_extract_recipe` ×2, `agent_extract_recipe_from_image`) had no error handling, unlike the URL-fetch call in the same function and the discovery-agent calls in `recipe_discovery.py` | Fix, make consistent | Wrapped all three in `try/except Exception` plus a falsy-return check; added a small `_extraction_error()` helper so every failure branch in the file renders through the same shape | `recipe_extract.py` | Agent raising and agent returning `None` both now render the preview partial's error state with a 200, for all three call sites, instead of an unhandled 500 |
| 9 | Whether `recipe_micro_partials.html` is genuinely dead code | Could not fully verify | None | — | See §6.1 in full — kept verbatim at the requester's instruction |

## 5. Files in Final State

| File | Lines | Contents |
|---|---|---|
| `_extraction_helpers.py` | 203 | Pure helpers: `_fetch_url_content`, `_parse_schema_org`, `_strip_html`. No route decorators, no FastAPI dependencies beyond `httpx`. |
| `recipe_extract.py` | 280 | Four route handlers (`extract_landing`, `extract_from_url`, `extract_from_file`, `confirm_extraction`) plus the new `_extraction_error()` helper. |
| `recipe_mutations.py` | 107 | Three route handlers (`rate_recipe`, `toggle_favorite`, `log_cook`), each returning a micro-partial. |
| `recipes.py` | 292 | Library/CRUD: `recipe_library`, `list_tags`, `suggest_ingredients`, `recipe_detail`, `create_recipe`, `update_recipe`, `delete_recipe`. |

## 6. Open Items Carried Forward

### 6.1 — `recipe_micro_partials.html` (Item 9), kept in full per the requester

> `recipe_micro_partials.html`'s content is a byte-for-byte concatenation
> of `recipe_rating.html`, `recipe_favorite.html`, and
> `recipe_cook_count.html` — including their individual header comment
> blocks (each still says `templates/partials/recipe_rating.html`, etc.,
> as if it were its own file). None of the three endpoints relocated into
> `recipe_mutations.py` (`rate_recipe`, `toggle_favorite`, `log_cook`)
> reference `"partials/recipe_micro_partials.html"` — each returns its
> own individual partial file. The best guess is this was a scratch file
> used to sketch all three micro-partials together during initial
> development, then the three were split out into their real files and
> this combined version was never deleted.
>
> This was checked only against the documents pasted into this
> conversation — not against the real repository, which was not
> available in this session. None of the templates shared
> (`recipe_extract.html`, `recipe_discover.html`, `recipe_detail.html`,
> `recipes.html`, `pantry.html`, `discovery_results.html`,
> `discovery_save_result.html`, `pantry_list.html`) reference
> `recipe_micro_partials.html`. That is a partial answer, not a real one,
> for two reasons: **`base.html` was not shared**, and every template in
> this domain extends it — per GOVERNANCE §3.2 it's exactly the kind of
> shared layout file where a stray `{% include %}` would live if one
> existed. And **other domains were not shared** — something outside
> `domains/recipes/` (a debug page, an admin view, another domain
> experimenting with a shared partial) could reference it, with no way to
> see that from what's in this conversation.
>
> So: **not proven dead, just not referenced anywhere visible from this
> session.** Before deleting it, run this against the real repo:
> ```bash
> grep -rn "recipe_micro_partials" internal_dataplayground/
> ```
> If that comes back empty, it's a safe standalone deletion — separate
> from WO#26, per GOVERNANCE §4.6's "no cleanup bundled into a
> migration" rule.

### 6.2 — Cross-references back into §4
Items 1 and 3 above (the `recipes.py` line count, and the rollback gap
for the two new files) are not further discussed here beyond §4's table
and §7/§8 below, where they're turned into concrete follow-up actions.

## 7. Rollback Instructions (Updated for This Session's Actual Diff)

The original WO's rollback instruction — "`git checkout` on every file
touched" — has two gaps once §4's changes are folded in:

1. It doesn't cover the two brand-new files. There's no prior commit for
   `git checkout` to restore from.
2. Because §4's fixes were applied to the **same files** as the Part A/B
   split in this session, a single `git checkout` on `recipe_extract.py`
   or `recipes.py` would revert **both** the relocation and the six
   hardening fixes at once — even if the reviewer only wants to undo one
   of the two.

**Full rollback to pre-WO#26 state:**
```bash
git checkout -- domains/recipes/routers/recipe_extract.py
git checkout -- domains/recipes/routers/recipes.py
git checkout -- main.py               # only if the suggested diff was applied
git rm domains/recipes/routers/_extraction_helpers.py
git rm domains/recipes/routers/recipe_mutations.py
```

**Recommended before merging:** split this session's output into two
commits so each is independently revertible, matching how the project's
own process (GOVERNANCE §4.5) says this should have been done in the
first place:
- **Commit 1 — "WO#26: split recipe_extract.py and recipes.py by
  responsibility"** — the pure relocation, matching §2's acceptance
  criteria exactly, with none of §4's fixes.
- **Commit 2 — "Post-migration hardening found during WO#26 review"** —
  items 2, 4, 5, 6, 7, 8 from §4, as their own diff, ticketed separately.

## 8. Follow-On Work Required After All Domain Migrations Are Completed

This section is written for whichever agent eventually closes out
GOVERNANCE.md §2.4's "Legacy Import Shims" process for the recipes
domain (and, if it's still open, for the other domains named in §3.3).
It assumes no prior context beyond this document and the repository
itself.

### 8.1 An ambiguity that has to be resolved first

Two parts of the provided project documentation appear to disagree about
whether the recipes domain (and several others) are actually "migrated"
in the sense GOVERNANCE.md cares about:

- **GOVERNANCE.md §3.3 ("Migration Debt Tracker")** lists `recipes (+
  pantry)` among domains "not yet moved into the `domains/` structure, as
  of this document," alongside `finance`, `journal`, `workout`, `media`,
  and `planning`.
- **The actual repository state** shows `domains/recipes/models.py`,
  `domains/recipes/routers/*`, and `domains/recipes/templates/*` already
  fully populated, and `main.py` tags the recipes router imports `# WO7`
  — i.e., already migrated, several work orders ago.

This is either (a) §3.3 going stale the moment WO#7 shipped and never
being updated, or (b) some distinction between "the domain folder exists"
and "the migration is formally closed" that isn't written down anywhere
else in GOVERNANCE.md. Given `main.py` also tags `finance` as WO5,
`journal` as WO6, `workout` as WO8, `media` as WO9, and `planning` as
WO10 — i.e., *every single domain* §3.3 lists as "not yet migrated"
already has a domain folder and a main.py registration — (a) is the far
more likely explanation, but it should be **confirmed against the real
repo**, not assumed, before any cleanup proceeds.

There's a second, compounding wrinkle: **GOVERNANCE.md §2.4** separately
states, under a `Status: historical/closed` heading, that WO#20 already
removed every remaining domain shim from root `models.py`, and WO#22
already deleted root `models.py` entirely, once a repo-wide grep
confirmed it had no real consumers left. Recipes is WO#7 — chronologically
well before both WO#20 and WO#22. If that closed status is accurate, root
`models.py` shouldn't exist anymore at all, which would mean the shim
described in `domains/recipes/models.py`'s own module docstring —
*"RECIPE MANAGER MODULE — moved to domains/recipes/models.py as part of
the domain-folder migration (Work Order #7). Re-exported from root
models.py so any other file still doing `from models import Recipe`
(etc.) keeps working unchanged."* — is describing a shim that was already
removed two work orders ago, and the docstring is simply stale.

**Concrete first step, before anything else in this section:**
```bash
test -f internal_dataplayground/models.py && echo "root models.py EXISTS" || echo "root models.py ABSENT"
grep -rn "from models import" internal_dataplayground/ --include="*.py" | grep -v "/domains/"
```
The result determines which of §8.2 or §8.3 applies.

### 8.2 If root `models.py` still exists

1. Confirm what recipes-domain names it currently re-exports. Based on
   `domains/recipes/models.py`'s public surface, expect something like:
   ```python
   # TODO: remove after all cross-references are updated
   from domains.recipes.models import (
       Ingredient, RecipeTag, Recipe, RecipeIngredient, PantryItem,
       IngredientCategory, RecipeSourceType, RecipeMealType,
       RecipeDifficulty, IngredientUnit,
       IngredientResponse, RecipeIngredientResponse, RecipeTagResponse,
       RecipeResponse, RecipeCreate, PantryItemResponse,
   )
   ```
2. Grep the whole repo for `from models import` (root-level, not
   `from domains.recipes.models import`) to find every remaining
   consumer.
3. Per GOVERNANCE §2.2, the only sanctioned cross-domain consumer should
   be `routers/dashboard.py`. If that's the only hit, update it to import
   directly from `domains.recipes.models`.
4. If anything else shows up, that's a cross-domain-import boundary
   violation predating this work order — file it as its own ticket per
   §4.5, don't fix it inline as part of this cleanup.
5. Once `dashboard.py` (or nothing) is the only consumer and has been
   updated, delete the recipes shim block from root `models.py`.
6. Repeat steps 1–5 for every other domain still holding a shim, per
   §3.3's tracker — once §8.1 has confirmed that tracker is actually
   accurate.
7. Once every domain's shim is gone, redo the repo-wide grep described in
   §2.4's WO#20/WO#22 precedent. If it comes back empty, delete root
   `models.py` entirely and mark §2.4 formally closed — updating its
   "Status" line if it isn't already, and fixing the document's own
   front-matter ("Last updated after Work Orders #1–4"), which is
   already stale relative to the document's own body (it discusses
   WO#18, #20, #22, #25 later on).

### 8.3 If root `models.py` no longer exists (WO#20/#22 already fully applied)

1. The recipes-domain shim was already swept up in that cleanup. The
   remaining work is documentation-only:
   - Rewrite the stale sentence in `domains/recipes/models.py`'s module
     docstring quoted in §8.1 — the re-export it describes no longer
     exists. Replace it with something like: *"the root-level re-export
     shim was removed in WO#20/#22 — this module is the sole source of
     these classes now."*
   - Confirm `routers/dashboard.py` already imports recipes classes from
     `domains.recipes.models` directly (it must, if root `models.py` is
     gone) — a confirmation step, not a change, but worth checking off
     explicitly rather than assuming.
2. Correct GOVERNANCE.md §3.3 to remove `recipes (+ pantry)` from the
   "not yet moved" list. If `finance`, `journal`, `workout`, `media`, and
   `planning` are similarly already reflected in `main.py` under their
   own WO tags, the entire §3.3 section is likely obsolete and should be
   reviewed for removal rather than edited domain-by-domain.
3. Fix GOVERNANCE.md's front-matter status line while already in the
   file, per the same reasoning as §8.2 step 7 — unrelated to recipes
   specifically, but noticed while reasoning through this section, and
   §6 (Amendment Process) treats exactly this kind of finding as fair
   game to fold in.

### 8.4 Two standing tickets from this work order, independent of the models.py question

1. **`recipe_micro_partials.html`** — needs the repo-wide grep described
   in §6.1 before any deletion decision.
2. **`recipes.py` line count** — 292 lines, past GOVERNANCE §1.2's
   250-line "should split before crossing 300" guidance. A natural
   follow-up split (out of scope for WO#26's Part B, which explicitly
   kept these functions together) is read vs. write, mirroring the
   `workout_plans.py` → `workout_plans_crud.py` /
   `workout_plan_ai_generator.py` precedent GOVERNANCE §1.2 itself cites:
   `recipe_library`, `recipe_detail`, `list_tags`, `suggest_ingredients`
   stay in `recipes.py`; `create_recipe`, `update_recipe`,
   `delete_recipe` move to a new `recipe_write.py`. This is a candidate
   for its own work order, using the standing template in GOVERNANCE
   §4.3.

## 9. Reviewer Sign-off Checklist

Mapped to GOVERNANCE.md §4.4's own review order:

1. **Every hard boundary respected?** Only files in Part A/B's SCOPE plus
   `main.py` were touched. The `main.py` edit is a single router
   registration addition, consistent with the WO's own carve-out
   ("touch it only if route registration ordering needs revisiting") —
   registering the new router is required for the split to function at
   all, not an unrelated improvement.
2. **Are ❌/⚠️ items genuinely out of the agent's control?** Yes — no
   `scripts/check_router_line_limits.py`, no live database, no live
   Gemini/agent implementations, no real Jinja2 templates were available
   in this session to run against.
3. **Does Notes surface anything needing its own ticket?** Yes — §8's
   two standing tickets, plus the models.py/GOVERNANCE §3.3 ambiguity in
   §8.1, none of which should be folded into this work order's own
   sign-off.
4. **Do the acceptance criteria that matter functionally actually pass?**
   Yes, per §2's table — all ✅ or ⚠️ with a stated, external reason, none
   ❌.
5. **(Specific to this document)** Confirm §4's nine items are being
   evaluated as a *separate* changeset from the WO#26 relocation proper,
   per GOVERNANCE §4.5, even though both landed in the same files in this
   session's output. Sign off on WO#26 (§1–§3) and the hardening pass
   (§4) as two distinct decisions, not one.
