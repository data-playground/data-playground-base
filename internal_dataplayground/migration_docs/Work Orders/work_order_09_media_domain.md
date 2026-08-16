# Work Order #9 — Domain Migration: `media`

*This domain has no external consumers at all — confirmed by reviewing
every other router in the codebase, including `dashboard.py`. This should
be the cleanest migration since WO#4 (`explorer`): no shim-verification
against a cross-domain reader is needed, because there isn't one. The main
complexity here is external: this domain depends on the `ml-service`
Docker container for embeddings/similarity, and has an environment-variable
feature toggle read once at import time.*

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
- **`services/ml_service_client.py`, `services/tmdb_service.py`, and
  `services/openlibrary_service.py` are explicitly OUT OF SCOPE and must
  NOT be moved.** They are shared external-API wrappers, same tier as
  `github_service.py`. None of them import `models`, so — unlike
  `recipe_service.py` in WO#7 — they need no internal edits either. Leave
  every import of them completely untouched.
- **`routers/media_recommend.py` reads an environment variable
  (`MEDIA_RECOMMEND_AI`) once, at module import time**, into a module-level
  constant `_USE_GEMINI`:
  ```python
  _USE_GEMINI = os.environ.get("MEDIA_RECOMMEND_AI", "false").lower() == "true"
  ```
  This means the toggle's value is fixed for the lifetime of the running
  process — it will NOT change if you flip the environment variable without
  restarting/reimporting. **This is expected, existing behavior, not a bug
  to fix.** When verifying this router post-move, be aware that a "the
  toggle didn't take effect" observation during testing is very likely just
  this known import-time-caching behavior, not a migration regression —
  confirm by checking whether the *same* environment variable value would
  have produced the same behavior pre-migration before treating it as a
  finding.
- `media_recommend.py`'s Gemini explanation layer (`_gemini_explain`) is
  another of the six known duplicate AI-client implementations tracked in
  GOVERNANCE.md §2.3. **Do not touch it, do not route it through any
  service layer.**

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
the result ⚠️ rather than ✅ or leaving it blank. (Note: the `ml-service`
container, TMDB API, OpenLibrary API, and Gemini explanation layer are all
external dependencies that cannot be exercised live in this environment —
verify each route reaches its respective client call correctly with a
mocked/stubbed response, mark those checks ⚠️ with this explanation, same
pattern as every prior work order's external-API caveats.)

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
`PREDEFINED_MOOD_TAGS` (module-level list constant — not a class, move it
as-is alongside the classes below), `MediaExternalSource`, `MediaType`,
`UserMediaStatus`, `RecommendationMediaType`, `StreamingService`,
`MediaItem`, `UserMedia`, `TVSeasonProgress`, `MediaRecommendation`,
`MediaItemResponse`, `UserMediaCreate`, `UserMediaUpdate`,
`UserMediaResponse`, `StreamingServiceResponse`

**Routers:**
- `routers/media.py`
- `routers/media_search.py`
- `routers/media_recommend.py`
- `routers/media_settings.py`

**Templates:**
- `templates/media.html`
- `templates/media_search.html`
- `templates/media_recommend.html`
- `templates/media_settings.html`
- `templates/media_rec_history.html`
- `templates/partials/media_card.html`
- `templates/partials/media_detail.html`
- `templates/partials/media_drawer_status.html`
- `templates/partials/media_rating.html`
- `templates/partials/media_seasons.html`
- `templates/partials/recommendations.html`
- `templates/partials/search_results.html`
- `templates/partials/streaming_service_list.html`

**Static:**
- `static/css/media.css`

**Core/config files to edit:**
- `models.py`
- `main.py`
- `core/templating.py`

**Not in scope, referenced only to confirm no breakage:** none. Unlike
every prior domain migration, no other file in the codebase — including
`routers/dashboard.py` — imports anything from this domain. This is a
required verification step (see ACCEPTANCE CRITERIA), not an assumption to
skip checking.

---

## STEPS

1. **Create `domains/media/models.py`.** Move `PREDEFINED_MOOD_TAGS`
   first (it's referenced by both the router and, implicitly, by template
   logic that iterates over it), then the four enums
   (`MediaExternalSource`, `MediaType`, `UserMediaStatus`,
   `RecommendationMediaType`), then `StreamingService`, `MediaItem`,
   `UserMedia`, `TVSeasonProgress`, `MediaRecommendation`, then the five
   Pydantic schemas — preserve current relative order. Import `Base` from
   `core.base_model`. All relationships (`MediaItem.user_media`,
   `UserMedia.media_item`, `UserMedia.season_progress`,
   `TVSeasonProgress.user_media`) are same-module string references — no
   special handling needed.

2. **In `models.py`:** delete the fifteen moved definitions (1 constant +
   4 enums + 5 classes + 5 Pydantic schemas) and replace with a re-export
   shim: `from domains.media.models import PREDEFINED_MOOD_TAGS,
   MediaExternalSource, MediaType, UserMediaStatus,
   RecommendationMediaType, StreamingService, MediaItem, UserMedia,
   TVSeasonProgress, MediaRecommendation, MediaItemResponse,
   UserMediaCreate, UserMediaUpdate, UserMediaResponse,
   StreamingServiceResponse`. Tag it `# TODO: remove after all
   cross-references are updated`.

3. **Move routers:**
   - `routers/media.py` → `domains/media/routers/media.py`
   - `routers/media_search.py` → `domains/media/routers/media_search.py`
   - `routers/media_recommend.py` → `domains/media/routers/media_recommend.py`
   - `routers/media_settings.py` → `domains/media/routers/media_settings.py`

   Update each file's model imports to pull from `domains.media.models`
   instead of `models`. Update each file's `templates =
   Jinja2Templates(directory="templates")` to `from core.templating import
   templates`. Leave every `from services.ml_service_client import ...`,
   `from services import tmdb_service, openlibrary_service`, and the
   `_USE_GEMINI` / `os.environ.get("MEDIA_RECOMMEND_AI", ...)` logic in
   `media_recommend.py` completely untouched.

4. **Move templates**, preserving the `partials/` subfolder structure, into
   `domains/media/templates/` per the SCOPE list above.

5. **Move `static/css/media.css`** to
   `domains/media/static/css/media.css`. Update every `<link
   rel="stylesheet" href="/static/css/media.css">` reference (present
   across all five media templates) to `/static/media/css/media.css`.

6. **Update `core/templating.py`'s `ChoiceLoader`** to add
   `domains/media/templates/` as an additional search root, alongside the
   roots already added in WO#1–8.

7. **In `main.py`:**
   - Update the four router imports/includes to their new paths (`from
     domains.media.routers import media, media_search, media_recommend,
     media_settings`).
   - **Preserve the existing include order and the comment explaining
     it** — `main.py` currently registers `media_search`, `media_recommend`,
     and `media_settings` before the general `media` router, with a
     comment noting `media` is the catch-all and must come last. Keep that
     exact relative ordering after the import path change.
   - Add the new static mount: `app.mount("/static/media",
     StaticFiles(directory="domains/media/static"), name="media_static")`.
     Register it **before** the general `/static` mount, per the ordering
     rule in GOVERNANCE.md §2.6.

---

## ACCEPTANCE CRITERIA

- [ ] `GET /media` renders identically — filter bar (type/status/service
  pills), media grid, empty state
- [ ] `GET /media/search` and `GET /media/search/query` render/return
  identically — movie/TV/book search type tabs, results list
  (mocked/stubbed TMDB/OpenLibrary calls per WORKING METHOD, mark ⚠️)
- [ ] `POST /media/search/add` still correctly finds-or-creates a
  `MediaItem` and `UserMedia`, returning the confirmation fragment
- [ ] `GET /media/recommend` renders identically — controls panel,
  liked-count warning states, history preview
- [ ] `POST /media/recommend/generate` reaches the ML service client and
  (if `MEDIA_RECOMMEND_AI` is enabled) the Gemini explanation layer
  correctly; mark ⚠️ per WORKING METHOD since neither can be exercised
  live here
- [ ] `GET /media/recommend/history` renders `media_rec_history.html`
  correctly
- [ ] `GET /media/settings` renders identically — streaming service toggle
  grid
- [ ] `POST /media/settings/subscriptions` still correctly bulk-updates
  subscription state and returns `streaming_service_list.html`
- [ ] `GET /media/{id}/detail`, `PATCH /media/{id}/status`, `PATCH
  /media/{id}/rate`, `PATCH /media/{id}/notes`, `POST
  /media/{id}/seasons/{season_number}`, and `DELETE /media/{id}` all still
  work and return their respective partials correctly
- [ ] `Base.metadata` table-identity check (method established in WO#1):
  same table count before/after, `models.MediaItem is
  domains.media.models.MediaItem`, `models.UserMedia is
  domains.media.models.UserMedia`, no `InvalidRequestError` on mapper
  configuration
- [ ] **Confirm zero external consumers**, matching the SCOPE claim above:
  `grep -r "from models import"` for each of the moved class/constant
  names across the repo returns **only** the shim's own lines in
  `models.py` — if you find any reference outside `models.py` and this
  domain's own files, treat that as a discrepancy from what this work
  order assumed and report it explicitly rather than silently reconciling
  it
- [ ] Confirm explicitly (git diff or direct statement) that
  `routers/dashboard.py` required **zero** changes for this migration

---

## For the next work order (not part of this one)

Per GOVERNANCE.md §3.3, **Work Order #10 = `planning`** (`weekly_plan.py` +
`intent.py`) is the last of the real domains, saved for last specifically
because it accumulates cross-domain references into every domain migrated
so far. Before drafting that work order, do a fresh reconciliation pass
across WO#5–9's "Not in scope" sections — `weekly_plan.py` alone is on
record as depending on: `Recipe`/`RecipeMealType`/`Ingredient`/`PantryItem`
(+ local `RecipeIngredient` import) from WO#7, `WorkoutPlan`/
`WorkoutPlanDay`/`WorkoutSession`/`WeightUnit` from WO#8, and
`journal.py`'s local `WeeklyPlanDay`/`WeeklyPlan`/`WeeklyPlanStatus` import
from WO#6 runs in the opposite direction (journal reaching into planning).
`weekly_plan.py` is also already past the 300-line rule (GOVERNANCE.md
§1.2) — Work Order #10 should treat splitting it as part of the same
change, not a follow-up, since a location-only move of an oversized file
just relocates the size problem rather than fixing it. `intent.py` is
small and clean by comparison and has no cross-domain complexity of its
own — it moves along with `weekly_plan.py` only because
`UserIntent.to_ai_context()` is consumed directly by planning's AI
generator calls, not because it needs the same scrutiny.

After Work Order #10, `dashboard` is the only domain intentionally left at
the top level (per GOVERNANCE.md §2.2) — at that point every "real"
migration on the backlog (GOVERNANCE.md §3.3) will be complete, and it will
be the right time to circle back to the shim-removal cleanup pass
(GOVERNANCE.md §2.4) across every domain at once, now that `dashboard.py`
is the only remaining consumer of any of them.
