# Work Order #5 — Domain Migration: `finance`

*First domain from the non-priority backlog (GOVERNANCE.md §3.3). Shape is
similar to `jobs` (WO#3) — four routers sharing one model set, one external
cross-domain consumer (`dashboard.py`) reading a single class
(`Transaction`) rather than the whole domain.*

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
- `finance_upload.py` calls out to the Gemini API (via the `google-genai`
  SDK, not the raw-REST pattern used elsewhere) for transaction
  categorization. **Do not touch this logic, do not attempt to route it
  through any AI service layer, and do not "fix" the fact that it's a
  different calling convention than the rest of the codebase** — this is
  known, tracked debt (GOVERNANCE.md §2.3) and explicitly out of scope for
  a location-only refactor.
- `templates/partials/account_options.html` exists in the current codebase
  but does not appear to be referenced by any router or template found
  during review (`finance_upload.html` builds its own inline `<select>`
  rather than including this partial). **Move it anyway** as part of this
  domain (it's finance-shaped content regardless of current usage), but
  explicitly verify and report whether anything actually includes it — do
  not delete it, do not assume it's dead without confirming.

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
the result ⚠️ rather than ✅ or leaving it blank. (Note: `finance_upload.py`'s
Gemini categorization call cannot be verified against the live API in this
environment — verify the route accepts a CSV, parses it, and reaches the
categorization call correctly with a mocked/stubbed response; mark that
specific check ⚠️ with this explanation, same pattern as the GitHub/ATS
caveats in WO#2 and WO#3.)

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
`AccountType`, `Category`, `Account`, `Transaction`, `AccountCreate`,
`AccountResponse`, `CategoryCreate`, `CategoryResponse`, `TransactionResponse`

**Routers:**
- `routers/finance_summary.py`
- `routers/finance_ledger.py`
- `routers/finance_upload.py`
- `routers/finance_settings.py`

**Templates:**
- `templates/finance.html`
- `templates/finance_ledger.html`
- `templates/finance_upload.html`
- `templates/finance_settings.html`
- `templates/partials/upload_result.html`
- `templates/partials/account_list.html`
- `templates/partials/category_list.html`
- `templates/partials/account_options.html` (see HARD BOUNDARIES note above)

**Static:**
- `static/css/finance.css`

**Core/config files to edit:**
- `models.py`
- `main.py`
- `core/templating.py`

**Not in scope, referenced only to confirm no breakage:**
- `routers/dashboard.py` (imports `Transaction` from `models` for the
  monthly income/expense/net summary card — must keep working via shim,
  same pattern as every prior domain)

---

## STEPS

1. **Create `domains/finance/models.py`.** Move `AccountType`, `Category`,
   `Account`, `Transaction`, and their four Pydantic schemas there
   verbatim, preserving current relative order. Import `Base` from
   `core.base_model`. Note: `Account.transactions` and
   `Transaction.account` reference each other via string class names
   within this same file — no special handling needed, consistent with the
   `Job`/`ApplicationLog` same-module case in WO#3.

2. **In `models.py`:** delete the nine moved class bodies and replace with
   a re-export shim: `from domains.finance.models import AccountType,
   Category, Account, Transaction, AccountCreate, AccountResponse,
   CategoryCreate, CategoryResponse, TransactionResponse`. Tag it `# TODO:
   remove after all cross-references are updated`.

3. **Move routers:**
   - `routers/finance_summary.py` → `domains/finance/routers/finance_summary.py`
   - `routers/finance_ledger.py` → `domains/finance/routers/finance_ledger.py`
   - `routers/finance_upload.py` → `domains/finance/routers/finance_upload.py`
   - `routers/finance_settings.py` → `domains/finance/routers/finance_settings.py`

   Update each file's model imports to pull from `domains.finance.models`
   instead of `models`. Update each file's `templates =
   Jinja2Templates(directory="templates")` to `from core.templating import
   templates`. Leave `finance_upload.py`'s `from database import get_db`
   and its `google.genai` import untouched — neither changes.

4. **Move templates**, preserving the `partials/` subfolder structure, into
   `domains/finance/templates/` per the SCOPE list above.

5. **Move `static/css/finance.css`** to
   `domains/finance/static/css/finance.css`. Update the `<link
   rel="stylesheet" href="/static/css/finance.css">` references inside
   `finance.html` and `finance_ledger.html` to
   `/static/finance/css/finance.css`. Check `finance_upload.html` and
   `finance_settings.html` for the same reference and update if present —
   if either page has no such link (relying on inline styles or `base.css`
   only), note that in your report rather than adding one.

6. **Update `core/templating.py`'s `ChoiceLoader`** to add
   `domains/finance/templates/` as an additional search root, alongside
   the roots already added in WO#1–4.

7. **In `main.py`:**
   - Update the four router imports/includes to their new paths (`from
     domains.finance.routers import finance_summary, finance_ledger,
     finance_upload, finance_settings`).
   - **Preserve the existing include order and the comment explaining
     it** — `main.py` currently registers the finance routers with a
     comment noting "specific prefixes before catch-all /finance summary."
     Keep that relative ordering after the import path change.
   - Add the new static mount: `app.mount("/static/finance",
     StaticFiles(directory="domains/finance/static"), name="finance_static")`.
     Register it **before** the general `/static` mount, per the ordering
     rule in GOVERNANCE.md §2.6.

---

## ACCEPTANCE CRITERIA

- [ ] `GET /finance` renders identically — KPI cards, category chart,
  recent transactions table, and both the "has data" and "empty month"
  states (test by checking a month with no transactions if possible, or by
  reviewing the conditional template logic matches pre-migration)
- [ ] `GET /finance/ledger` renders identically — filters, summary row,
  transaction table, inline category editor markup
- [ ] `GET /finance/upload` renders identically — account select, drop
  zone, column mapper (hidden until a file is parsed client-side, so just
  confirm the DOM structure is present)
- [ ] `GET /finance/settings` renders identically — account list panel,
  category list panel, both add-forms
- [ ] `POST /finance/accounts` and `DELETE /finance/accounts/{id}` still
  return `account_list.html` correctly
- [ ] `POST /finance/categories` and `PATCH
  /finance/categories/{id}/toggle` still return `category_list.html`
  correctly
- [ ] `PATCH /finance/transactions/{id}/category` still returns the
  correct `<td>` fragment with the updated category badge
- [ ] `POST /finance/upload` — verify it accepts a CSV, parses rows, and
  reaches the categorization call correctly; mark ⚠️ per the WORKING
  METHOD note above since live Gemini calls aren't available in this
  environment. Confirm error paths (missing column, empty CSV) still
  return `html_error()` fragments correctly, since those don't require the
  live API.
- [ ] `Base.metadata` table-identity check (method established in WO#1):
  same table count before/after, `models.Transaction is
  domains.finance.models.Transaction`, `models.Account is
  domains.finance.models.Account`, no `InvalidRequestError` on mapper
  configuration
- [ ] `routers/dashboard.py`'s existing `Transaction` import (via `from
  models import ...`) still resolves, and `/dashboard`'s finance summary
  card (income/expenses/net for the current month) renders correctly
- [ ] `grep -r "from models import"` for each of the nine moved class names
  across the repo returns only the shim's own lines in `models.py` plus
  `routers/dashboard.py`'s `Transaction` import — nothing else should need
  updating
- [ ] Confirm and report whether `templates/partials/account_options.html`
  is actually included/referenced anywhere in the codebase (see HARD
  BOUNDARIES) — move it regardless, but the usage finding itself is a
  required part of this report

---

## For the next work order (not part of this one)

Per GOVERNANCE.md §3.3's suggested batching, **Work Order #6 = `journal`**
is a good next step — a single router (`routers/journal.py`) paired
conceptually with the `life_os_weekly_synthesis` DAG (which, per the
standing DAG-relocation rule, stays out of scope and untouched). It has a
notable privacy constraint documented directly in its model/router
docstrings (`content`, `gratitude`, `challenges` fields must never be sent
to any external AI call) — worth carrying into that work order's HARD
BOUNDARIES section explicitly, since it's a correctness-and-privacy
constraint, not just a style preference, and should be called out the same
way the explorer domain's `BLOCKED_PATTERN` security logic was in WO#4.
