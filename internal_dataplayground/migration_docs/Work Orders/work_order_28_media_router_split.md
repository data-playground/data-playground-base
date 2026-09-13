# Work Order #28 — Media Domain Router Split (2 files: `media_recommend.py`, `media.py`)

---

## ROLE
You are a senior refactoring engineer splitting oversized router files by
responsibility. Location-and-organization refactor only — no behavior
change, no schema change.

## HARD BOUNDARIES
- No endpoint's URL path, HTTP method, request/response shape, or template
  name changes.
- Do **not** touch the `_USE_GEMINI` module-import-time environment-variable
  read in `media_recommend.py`, or the `asyncio.to_thread(...)` wrapping
  around the synchronous Gemini call inside `_gemini_explain` — both are
  deliberate, documented behavior (see that function's own comment about
  not blocking the event loop) and unrelated to this split.
- Only touch the files named in each Part's SCOPE.

## HANDLING PRE-EXISTING BUGS
Don't fix, reproduce against baseline, report under Notes, mark ⚠️ not ❌.

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

## Part A — `media_recommend.py`

### SCOPE
- `domains/media/routers/media_recommend.py` (split)
- New: `domains/media/routers/_recommend_pipeline.py`

### STEPS
1. Move `_gemini_explain` and `_format_ml_only` — the scoring/explanation
   internals of the recommendation pipeline — into
   `_recommend_pipeline.py`.
2. Keep `recommendations_page`, `generate_recommendations`,
   `recommendation_history` in `media_recommend.py`.
3. See HARD BOUNDARIES — do not touch `_USE_GEMINI` or the
   `asyncio.to_thread` wrapping while moving `_gemini_explain`.

---

## Part B — `media.py`

### SCOPE
- `domains/media/routers/media.py` (confirm-then-maybe-split)
- New (only if needed): `domains/media/routers/media_seasons.py`

### STEPS
1. This file is the more borderline of the two — confirm the real current
   count via the lint script before doing anything.
2. If still over 300: the cleanest available split is moving
   `update_season_progress` (the one TV-season-specific endpoint) into
   `media_seasons.py`, alongside the season-progress query logic that's
   otherwise inline. Everything else (`media_board`, `media_detail`,
   `add_to_list`, `update_status`, `update_rating`, `update_notes`,
   `remove_from_list`) stays in `media.py`.
3. If already under 300: skip, report as not needed.

---

## ACCEPTANCE CRITERIA (both parts)
- [ ] All resulting files under 300 lines (or Part B explicitly confirmed
  not needed, with the real line count reported).
- [ ] `GET /media`, `GET /media/{id}/detail`, `POST /media`, `PATCH
  /media/{id}/status`, `PATCH /media/{id}/rate`, `PATCH /media/{id}/notes`,
  `DELETE /media/{id}` unchanged.
- [ ] `POST /media/{id}/seasons/{season_number}` unchanged if moved.
- [ ] `GET /media/recommend`, `POST /media/recommend/generate`, `GET
  /media/recommend/history` unchanged; `MEDIA_RECOMMEND_AI` toggle
  behavior unaffected (same import-time-caching caveat as WO#9 — don't
  mistake stale toggle state for a regression).

## For the next work order (not part of this one)
Can run fully in parallel with every other Track C work order, Track A,
Track B, and Track D.
