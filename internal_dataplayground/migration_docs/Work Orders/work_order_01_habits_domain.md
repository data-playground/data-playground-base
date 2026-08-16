# Work Order #1 — Domain Migration: `habits`

---

## Title
`POST /habits/log` and `DELETE /habits/log` return 500 — habit_card.html 
template context mismatch

## Severity
High — breaks the primary daily-use interaction on the Habits page (checking 
a habit on/off). Discovered during Work Order #1 (habits domain migration), 
confirmed present in the pre-migration baseline — not a regression from that 
work.

## Location
- `routers/habits.py` (now `domains/habits/routers/habits.py` post-migration)
  — `log_habit()` and `unlog_habit()` handlers
- `templates/partials/habit_card.html` (now `domains/habits/templates/partials/habit_card.html`)

## Root Cause
Both handlers build a `view` dict via `_build_habit_view()`:
```python
view = await _build_habit_view(db, habit, today, grace_period, today_logged_ids)
```
and then spread it directly into the template context:
```python
return templates.TemplateResponse(
    "partials/habit_card.html",
    {"request": request, **view, "today": today},
)
```
Spreading `**view` promotes its keys (`habit`, `today_logged`, `streak`) to 
top-level context variables. But `partials/habit_card.html` expects a single 
`view` object and references it as such throughout:
```jinja
<div class="habit-card {% if view.today_logged %}completed{% endif %}" ...>
    ...
    {% if view.streak > 0 %}
        <span class="habit-streak-count">{{ view.streak }}</span>
    ...
    {% if view.today_logged %}
    <button class="habit-toggle checked" ...>
```
Since `view` itself is never in the context (only its spread-out contents 
are), Jinja2 raises `UndefinedError: 'view' is undefined` and the request 
fails with a 500.

Note: `habits.html`'s main page loop passes `view` correctly (`{% include 
"partials/habit_card.html" %}` inside a `{% for view in habit_views %}` 
block, where `view` is naturally in scope) — that's why the initial page 
load works fine and only the HTMX log/unlog round-trip is broken. This masks 
the bug from casual testing.

## Confirmed Impact
- The underlying database write succeeds in both cases (log and unlog) — 
  confirmed via `/habits/progress` counts changing correctly across repeated 
  calls.
- Only the HTML fragment re-render fails, so the user sees an HTMX error / 
  broken swap rather than the updated card, even though their action was 
  recorded.

## Fix
In `routers/habits.py`, change both `log_habit()` and `unlog_habit()` to 
nest `view` instead of spreading it, matching what the template expects:
```python
return templates.TemplateResponse(
    "partials/habit_card.html",
    {"request": request, "view": view, "today": today},
)
```
This is a one-line change per handler (2 lines total). No template changes 
needed — `habit_card.html` already expects `view` as a nested object; it's 
the router that's inconsistent with its own template.

## Suggested Verification
1. Load `/habits`, confirm at least one active habit is present.
2. Click a habit's toggle to mark it done → should swap in the updated card 
   (checked state, streak incremented) with no console/network error.
3. Click it again to un-mark → should swap back to unchecked, streak 
   recalculated, no error.
4. Confirm `/habits/progress` count reflects each toggle correctly (this 
   part already works and should be unaffected).
5. Re-run Work Order #1's acceptance criterion #3 against this fix to close 
   out that ⚠️ as ✅.

## Scope Note
This ticket is standalone and does not require touching the domain-folder 
migration structure — the fix applies identically whether it's made against 
the pre-migration `routers/habits.py` or the post-migration 
`domains/habits/routers/habits.py`, whichever is current at the time this is 
picked up.