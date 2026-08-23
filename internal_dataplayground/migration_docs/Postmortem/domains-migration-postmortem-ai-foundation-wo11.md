# AI Service Layer — Postmortem (WO#11) & Roadmap (GOVERNANCE.md §2.3)

**Status:** Living document. Update after every AI Service Layer work order —
don't let it drift the way the six duplicate implementations it replaces did.

**How to use this document:** Section 1 is the postmortem for the work
completed so far (WO#11 only). Section 2 is the inventory of what's left.
Section 3 is the part to read most carefully if you're an agent picking up
any future work order in this initiative — it's the checklist for what
must happen *after every individual migration is done*, so nothing gets
silently left half-finished the way distributed refactors often do.

---

## 1. Postmortem — Work Order #11

### 1.1 What shipped

- `services/ai/keys.py` — `get_provider_key(provider)`, centralized key
  lookup for `"gemini"` / `"groq"` / `"cerebras"`.
- `services/ai/base.py` — `post_with_retry(url, payload, retries, *,
  provider_name, resource_label)`, the generic 429/503 retry loop
  extracted from `gemini_client.py`.
- `services/ai/providers/gemini.py` — `MODEL_FLASH`, `MODEL_FLASH_LITE`,
  `call_gemini_text()`, `call_gemini_json()`.
- `services/ai/__init__.py` — re-exports the above four names at the
  package top level.
- `airflow/agents/job_agents.py` — one-line import change, verified to be
  the *only* line that changed.
- `airflow/agents/gemini_client.py` — deleted, post-verification.

### 1.2 Key decisions and why

| Decision | Rationale |
|---|---|
| Keep `services/ai/base.py`'s retry contract Gemini-shaped (429/503 only, fixed backoff schedule) rather than trying to anticipate Groq/Cerebras | Only one real provider existed at migration time. Cerebras's actual retry logic (`blog_agents.py::_cerebras`) is materially different — `Retry-After` header handling, a 75/150/300/600s schedule, and a `(content, remaining_tokens)` return tuple. Guessing at a shared contract from one example risked building an abstraction nothing actually fits. |
| Defer `call_ai_text()` / `call_ai_json()` (the provider-agnostic wrapper) entirely | Same reasoning — GOVERNANCE.md §2.3's target interface needs ≥2 real providers to generalize from correctly. Building it now would mean guessing at what needs to be configurable (model name? temperature? JSON-mode support varies — Gemini has native schema enforcement, Groq/Cerebras don't). |
| Route `services/ai/keys.py` through `gcp_secrets.get_key()` rather than direct `os.environ.get()` | Initially deferred in WO#11 because `gcp_secrets.py`'s source wasn't available in that work order's scope. Once provided (see §1.4), traced that `get_key("GEMINI_API")`'s env-var branch is byte-for-byte identical to the old lookup in every real deployment (docker-compose always injects `GEMINI_API`). Updated after WO#11's initial delivery — see §1.4. |
| Re-export `MODEL_FLASH_LITE` at `services/ai/__init__.py`'s top level, not just in `providers/gemini.py` | Keeps `job_agents.py`'s migration a true single import line (`from services.ai import call_gemini_json, MODEL_FLASH_LITE`) instead of two imports from two paths for one logical change. |
| Delete `gemini_client.py` only after acceptance criteria passed | Rollback safety — a deleted file mid-verification is a much worse failure mode than a lingering unused one. |

### 1.3 Verification approach (reusable — copy this pattern for every future migration)

Live API keys aren't available in this environment, so equivalence was
proven mechanically instead of by inspection:

1. Captured the **pre-migration** behavior of the file being replaced
   (`gemini_client.py`) by running its real function against a mocked
   `requests.post`, recording: exact request URL, exact JSON payload,
   sleep-call arguments on a simulated 429, sleep-call arguments on a
   simulated 503, and the exact `RuntimeError` message text on exhausted
   retries.
2. Ran the **post-migration** code (`services.ai` + the edited caller)
   through the identical scenarios with the identical mocks.
3. Diffed the two captures programmatically (not eyeballed) — a clean
   diff was required before touching anything further (like deleting the
   old file).
4. Ran the **real, edited caller file** (`job_agents.py`, not a stand-in)
   end-to-end through the new package as a final sanity pass, plus a
   grep sweep (`grep -rn "gemini_client"`) confirming zero remaining
   import references anywhere in the repo.

**Recommendation:** every future migration in this initiative (WO#12
onward) should produce the same four artifacts — baseline capture,
migrated capture, programmatic diff, grep sweep — before deleting the
file it replaces. It's cheap, and it's the only way to actually justify
the HARD BOUNDARIES claim of "no behavior change to the actual API
calls" rather than asserting it.

### 1.4 Deviation from the literal WO#11 text (documented, not silent)

WO#11's Step 1 explicitly allowed *either* routing `keys.py` through
`gcp_secrets.get_key()` *or* falling back to direct `os.environ.get()`,
depending on whether `gcp_secrets.py`'s actual behavior could be confirmed
compatible — and instructed to note the discrepancy if not. At the time
WO#11 was delivered, `gcp_secrets.py`'s source wasn't in scope, so the
safer direct-env-var path was used, with a Note flagging the follow-up.

`gcp_secrets.py`'s source was provided immediately after delivery. Traced
and confirmed:

```python
def get_key(secret_name: str) -> str:
    env_key = secret_name.upper().replace("-", "_")   # "GEMINI_API" -> "GEMINI_API" (no-op)
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val
    # else: live GCP Secret Manager call for a secret literally named `secret_name`
```

`get_key("GEMINI_API")`'s env-var branch is identical to the old
`os.environ.get("GEMINI_API")` lookup, and that branch is what executes
in every real deployment (`docker-compose.yml` always injects
`GEMINI_API` into both the `web` and `airflow-*` containers). `keys.py`
was updated to call `gcp_secrets.get_key()`, re-verified against the same
four-artifact process in §1.3, confirmed identical in the case that
matters.

**One real, intentional behavior change survives, and is documented
in `keys.py`'s own docstring:** if `GEMINI_API` were ever unset, the old
code returned `None` silently (producing a broken request later); the new
code attempts a live Secret Manager call instead (fails loud, or
succeeds with a real fallback secret). This only diverges in an edge case
that doesn't occur given how the app is actually deployed, but it's a
real difference and it's flagged rather than absorbed.

### 1.5 Findings surfaced (not acted on — out of WO#11's scope, listed for future work orders)

**⚠ Cross-file coupling: `life_os_weekly_synthesis.py` imports a private
function from `blog_agents.py` directly.**

```python
# airflow/dags/life_os_weekly_synthesis.py, inside task_generate_synthesis():
from agents.blog_agents import _gemini_flash
...
synthesis_text = _gemini_flash(system_prompt, prompt)
```

This is a DAG reaching into another agent module's *private*
(underscore-prefixed) helper — not through any exported interface. This
matters specifically for whichever work order migrates `blog_agents.py`:
deleting `_gemini_flash` from that file (the way `gemini_client.py` was
deleted in WO#11) will silently break the Weekly Synthesis DAG's Gemini
fallback path *at runtime only* — the import is inside the task
function body, so it won't surface as an import error at DAG-parse time,
only when `task_generate_synthesis` actually executes. `_gemini_flash(system,
prompt)`'s signature matches `services.ai.call_gemini_text(system, prompt,
model=..., retries=...)` positionally, so the fix is likely a one-line
swap in `life_os_weekly_synthesis.py` too — but that DAG file isn't in
`blog_agents.py`'s own migration scope by default, so it needs to be
added to that work order's SCOPE explicitly or it'll be missed the same
way this document is trying to prevent.

**⚠ Possible discrepancy in the "how many duplicate implementations" count.**
WO#11's own text says both things in different places:
- Preamble: `gemini_client.py` is "the cleanest of the **six** duplicate
  implementations" (implying 6 total, 5 remaining after WO#11).
- Same preamble, one sentence later: "the remaining **five** duplicate
  implementations tracked in GOVERNANCE.md §2.3."
- HARD BOUNDARIES lists exactly **six** files/call-sites to leave
  untouched: `blog_agents.py`, `recipe_agents.py`, `weekly_agents.py`,
  `workout_plans.py`'s `_call_gemini_for_plan`, `media_recommend.py`'s
  `_gemini_explain`, `finance_upload.py`'s Gemini SDK call.

Either the "five" is a drafting slip, or one of the six listed items
doesn't actually count as a separate "implementation" for some reason not
stated in the WO text (e.g., maybe two of them were expected to share one
migration). This document treats all six as distinct migration targets
(§2) until someone with access to GOVERNANCE.md's actual §2.3 language
confirms otherwise — worth reconciling there directly.

**⚠ `recipe_agents.py` has a fourth call shape not yet represented in
`services/ai/providers/gemini.py`.** `_gemma()` calls
`gemma-4-31b-it` via the same Gemini REST endpoint shape as
`_gemini_flash`/`_gemini_flash_json`, but with no `systemInstruction`
field (the module docstring notes Gemma models don't support it — the
system context is prepended into the user prompt instead) and a JSON
`responseMimeType` but no `responseSchema`. `services/ai/providers/gemini.py`
currently only exposes `call_gemini_text` (no schema) and `call_gemini_json`
(schema required). Migrating `recipe_agents.py` will need either a third
function (`call_gemini_json` with an optional/omittable schema) or a
`MODEL_GEMMA` constant plus a signature tweak — decide when that WO starts,
don't guess now.

**⚠ `finance_upload.py`'s "Gemini SDK call" is architecturally different
from every other Gemini caller migrated or scoped so far.** Every other
Gemini call site (`gemini_client.py`, `recipe_agents.py`, `blog_agents.py`,
`weekly_agents.py`) uses raw REST via `requests`. WO#11's HARD BOUNDARIES
text specifically calls `finance_upload.py`'s call a "Gemini **SDK**
call" — distinct language. `google-genai==1.66.0` is installed in both
`requirements.txt` and `Dockerfile.airflow` but isn't imported anywhere
in any file reviewed so far — strongly suggesting `finance_upload.py` is
the one actual consumer of that SDK, and everyone else's install of it is
currently dead weight. Not confirmed — `finance_upload.py`'s content
hasn't been reviewed in this context. Flagged for confirmation in §2 and
addressed in §3's dependency-audit item.

**⚠ Three of the six duplicate call sites live in FastAPI routers, not
`airflow/agents/` modules.** `workout_plans.py`, `media_recommend.py`,
and `finance_upload.py` are routers (`main.py` confirms: `from routers
import ... workout_plans`, `from routers import ... media_recommend`,
`from domains.finance.routers import ... finance_upload`) — meaning their
AI calls happen synchronously inside what may be `async def` FastAPI
route handlers. `services/ai/base.py::post_with_retry` is a blocking,
synchronous function (plain `requests`, `time.sleep`). If any of these
three routers call it from an `async def` handler, that blocks the event
loop for the duration of the call + any retry sleeps — same
characteristic the pre-migration code already has (not a regression
introduced by migrating), but worth surfacing explicitly per WO#11's own
"handling pre-existing bugs" policy: confirm, don't silently fix,
report under that WO's Notes.

---

## 2. Remaining migration inventory (WO#12 and beyond)

| # | File | Provider(s) / functions | Consumers (confirmed) | Notes |
|---|---|---|---|---|
| 1 | `airflow/agents/recipe_agents.py` | Gemini — `_gemini_key`, `_gemini_flash`, `_gemini_flash_json`, `_gemma` (Gemma model, different payload shape — see §1.5) | `services/recipe_service.py` (`agent_normalize_ingredients`); presumably `routers/recipe_extract.py`, `routers/recipe_discovery.py` (not reviewed in this context) | **Recommended next (WO#12)** — structurally closest to `gemini_client.py`. Validates the pattern a second time before Groq/Cerebras. |
| 2 | `airflow/agents/blog_agents.py` | Gemini (`_gemini_flash`, `_gemini_flash_json`) + Groq (`_groq_llama`) + Cerebras (`_cerebras`) — three provider shapes in one file | `life_os_blog_scout.py`, `life_os_blog_creator.py`, `life_os_blog_finalizer.py`, `life_os_idea_expander.py`, `life_os_readme_writer.py`, `life_os_code_narrate.py`, `life_os_code_comment.py`, `life_os_code_improve.py`, **and `life_os_weekly_synthesis.py` via the private `_gemini_flash` import — see §1.5, must be added to this WO's scope explicitly** | Largest, most consumed, most architecturally divergent file in the initiative. First real test of Groq's and Cerebras's shapes against `services/ai/base.py`. Do this only after `recipe_agents.py` (WO#12) has validated the simple case. |
| 3 | `airflow/agents/weekly_agents.py` | Gemini only — `_gemini_flash_json` (no free-text variant used) | Presumably `routers/weekly_plan.py` (not reviewed) | Simplest remaining file — one function, one shape. Good candidate for WO#13/14 regardless of `blog_agents.py`'s outcome. |
| 4 | `routers/workout_plans.py` — `_call_gemini_for_plan` | Unknown — not reviewed in this context | The router itself; likely `templates`/frontend calling its endpoints | **Content not available in current context.** First step of this WO must be reading the actual file (same as WO#11 did for `job_agents.py`) before scoping anything. Confirm sync/async call-site concern from §1.5. |
| 5 | `routers/media_recommend.py` — `_gemini_explain` | Unknown — not reviewed in this context | The router itself | Same caveat as #4. |
| 6 | `domains/finance/routers/finance_upload.py` | Unknown — described only as a "Gemini **SDK** call" (`google-genai`), not REST | The router itself | Same caveat as #4, plus the SDK-vs-REST architecture question from §1.5 — this may not fit `services/ai/providers/gemini.py`'s current REST-only shape without adding SDK support to it. |

**Read `docs/note:`** items 4–6 are described from WO#11's HARD
BOUNDARIES text only — nobody working this initiative has actually
opened those three files yet. Do not assume their shape; confirm first.

---

## 3. Post-All-Migrations Consolidation (do this only after every row in §2 is migrated and verified)

This section is the actual "what happens after the individual migrations
are done" checklist. Frame it as the spec for a final work order —
call it **WO-Final** below — once items 1–6 in §2 are all complete.
None of this should start before then; doing it early risks reversing
work that a still-in-flight migration depends on.

### 3.1 Dead-code sweep — delete every now-unused per-file helper

Once all six original callers route through `services/ai`, the following
private functions should have zero remaining callers and should be
deleted, mirroring exactly what WO#11 did to `gemini_client.py`:

- `recipe_agents.py`: `_gemini_key`, `_gemini_flash`, `_gemini_flash_json`, `_gemma`
- `blog_agents.py`: `_gemini_key`, `_gemini_flash`, `_gemini_flash_json`, `_groq_key`, `_groq_llama`, `_cerebras_key`, `_cerebras`
- `weekly_agents.py`: `_gemini_key`, `_gemini_flash_json`
- `workout_plans.py`, `media_recommend.py`, `finance_upload.py`: whatever their equivalents turn out to be (unknown until those WOs run — see §2)

**Verification:** repeat WO#11's grep pattern, generalized:
```
grep -rn "_gemini_key\|_gemini_flash\|_gemini_flash_json\|_gemma(\|_groq_key\|_groq_llama\|_cerebras_key\|_cerebras(" \
  --include="*.py" .
```
A clean sweep should return zero hits outside `services/ai/` itself.
Also re-run the broader version of WO#11's key-lookup sweep:
```
grep -rn 'os\.environ\.get("GEMINI_API")\|os\.environ\.get("GROQ_API")\|os\.environ\.get("CEREBRAS_API")' \
  --include="*.py" .
```
Any hit outside `services/ai/keys.py` (or, transitionally, a file whose
migration WO hasn't landed yet) means a caller was missed.

### 3.2 Resolve the `life_os_weekly_synthesis.py` coupling explicitly

Per §1.5 — this must happen as part of (or immediately after) the
`blog_agents.py` migration, not deferred to WO-Final, since deleting
`_gemini_flash` without it breaks a DAG. Listed again here because
WO-Final's dead-code sweep (§3.1) is exactly the step that would surface
this if it were somehow missed earlier — treat a leftover `_gemini_flash`
import as a hard blocker for WO-Final, not something to route around.

### 3.3 Design and add the provider-agnostic wrapper

Now that Gemini, Groq, and Cerebras all have real `services/ai/providers/`
implementations, design `call_ai_text()` / `call_ai_json()` in
`services/ai/__init__.py` per GOVERNANCE.md §2.3's original target
interface — informed by the three real shapes now in hand, not guessed.
Specific open questions to resolve at this point (deliberately not
answered here — that's the point of deferring):

- Cerebras's `_cerebras()` returns `(content, remaining_tokens)`, not
  just `content`. Does the wrapper always return just content (dropping
  the token-budget signal for callers like `life_os_code_improve.py`
  that currently use it to throttle), or does it return a richer object
  (e.g. a small dataclass/NamedTuple) uniformly across all providers even
  though Gemini/Groq have no equivalent field to populate?
- Groq's current caller has *no retry logic at all*. Does the wrapper add
  retry uniformly (a behavior change, however small) or leave
  provider-specific retry behavior as an opt-in parameter?
- Gemini has native JSON-schema enforcement (`responseSchema`); Groq and
  Cerebras don't — they rely on prompt instructions + `_safe_json()`-style
  fence-stripping instead (see `recipe_agents.py::_safe_json`). Does
  `call_ai_json()` normalize this gap (e.g., always apply schema-stripping
  parsing as a fallback even for Gemini), or does it document the
  reliability difference and let callers choose their provider knowing that?
- Does `services/ai/base.py::post_with_retry`'s signature get extended to
  cover Cerebras's `Retry-After`-aware, differently-scheduled retry logic,
  or does Cerebras keep a separate `services/ai/base_cerebras.py`-style
  helper because forcing one shared function turned out to serve neither
  provider well? Make this call now, with three real examples in hand,
  rather than in WO#11 with only one.

### 3.4 Centralize the routing documentation

Every migrated file currently carries its own hand-maintained "MODEL
ROUTING" ASCII table in its module docstring (see `blog_agents.py`'s and
`recipe_agents.py`'s docstrings for the fullest examples) — these are
already slightly inconsistent in format from file to file and will drift
further as more migrations land. Once all six are done:
- Pick one canonical location for the routing table — a
  `services/ai/routing.py` registry module (agent name → provider →
  model, one row per agent) is a natural fit and can be imported by
  documentation generation later, vs. a plain `services/ai/README.md`
  which is easier to write but not machine-readable.
- Replace each file's full copy of the table with a one-line pointer to
  the canonical source.
- Keep the *rationale* prose (why Gemini vs. Groq vs. Cerebras for a
  given agent) — that's genuinely useful context, just don't duplicate
  the raw routing table alongside it six times.

### 3.5 Dependency audit — `requirements.txt` / `Dockerfile.airflow`

Once `finance_upload.py`'s migration (§2, item 6) has confirmed whether
its Gemini SDK usage was normalized to REST or kept as-is:
- If normalized to REST (recommended for consistency with every other
  provider call in the app): drop `google-genai==1.66.0` from both
  `requirements.txt` and `Dockerfile.airflow` — nothing else in the repo
  imports it.
- If kept SDK-based: `services/ai/providers/gemini.py` needs a second
  call shape (SDK, not just REST), and `google-genai` stays a genuine
  dependency — document why directly in that provider module so nobody
  "cleans it up" by mistake later.
- `cerebras-cloud-sdk` (in `Dockerfile.airflow` only, not
  `requirements.txt`) is confirmed genuinely used by `blog_agents.py`'s
  `_cerebras()` today — this one stays regardless, just confirm it moves
  cleanly into whatever `services/ai/providers/cerebras.py` becomes.
- No Groq SDK is installed anywhere — `_groq_llama()` calls Groq's REST
  API directly via `requests`. Confirm this stays true in
  `services/ai/providers/groq.py`, or note explicitly if the migration
  introduces the `groq` PyPI package instead.

### 3.6 `GOVERNANCE.md` §2.3 update

Mark the initiative complete. Replace whatever "six duplicate
implementations" (or "five" — see §1.5's flagged discrepancy; resolve it
here) language currently exists with a pointer to `services/ai/` as the
single source of truth for AI provider calls going forward, and a rule
that any *new* AI-calling code must be added there directly rather than
starting a seventh private implementation. Link this document (or its
final-state successor) for institutional memory of how the migration
actually went, not just that it happened.

### 3.7 The `models.py` question

Flagging this directly rather than silently either acting on it or
ignoring it: as of everything reviewed for WO#11 and this document,
**`models.py`** (the SQLAlchemy ORM / Pydantic schema definitions file)
contains no reference to `gemini_client`, any provider name, any API key,
or `services.ai` — it's purely data-model code, and nothing in this
initiative's scope currently touches it. If there's a specific reference
in mind there, it'd help to know what it is (a different file, or a part
of `models.py` not shown in this context) — but based on what's been
reviewed, there's nothing to remove from `models.py` for this initiative.

That said — keep a generic version of this check in WO-Final regardless:
**do a repo-wide search for AI-provider references outside `services/ai/`
and the six known caller files**, not just the specific files already
tracked in §2. The six-file inventory in §2 is built from what WO#11's
HARD BOUNDARIES text named plus what's been directly reviewed — it's
possible something outside that list (a template, a config file, a
domain module not yet reviewed) also has a stray reference that hasn't
surfaced yet. A broad closing sweep catches that regardless of whether
`models.py` specifically turns out to be relevant.

### 3.8 Regression pass across every consumer, not just the file just migrated

WO#11's acceptance criteria checked that `job_agents.py`'s two consuming
DAGs needed zero changes. WO-Final should re-run that style of check
**collectively** across every DAG/router that transitively imports any
migrated agent module, now that they all share `services/ai/base.py`'s
retry logic — a shared dependency means a bug in it now has blast radius
across every agent, not just one. At minimum, smoke-import every DAG file
that references any of the six migrated modules:

`life_os_blog_scout.py`, `life_os_blog_creator.py`, `life_os_blog_finalizer.py`,
`life_os_idea_expander.py`, `life_os_readme_writer.py`, `life_os_code_narrate.py`,
`life_os_code_comment.py`, `life_os_code_improve.py`, `life_os_weekly_synthesis.py`,
`life_os_job_scout.py`, `life_os_job_scout_ats.py`, `life_os_staging_promoter.py`
(uses `job_agents.py` indirectly), plus whichever routers own items 4–6
in §2 once their content is known.

### 3.9 WO-Final acceptance criteria (draft — refine once §2's items are actually done)

- [ ] Every function listed in §3.1 confirmed deleted; grep sweeps in §3.1 return zero unexpected hits
- [ ] `life_os_weekly_synthesis.py`'s Gemini call confirmed routed through `services.ai`, not a private `blog_agents` import
- [ ] `call_ai_text()` / `call_ai_json()` exist in `services/ai/__init__.py`, with the open questions in §3.3 resolved and documented (not left ambiguous)
- [ ] Every migrated file's docstring routing table replaced with a pointer to the canonical location chosen in §3.4
- [ ] `requirements.txt` / `Dockerfile.airflow` dependency audit (§3.5) complete, with `google-genai`'s fate explicitly decided and documented
- [ ] `GOVERNANCE.md` §2.3 updated per §3.6, including reconciling the five-vs-six discrepancy from §1.5
- [ ] Repo-wide sweep per §3.7 finds no AI-provider references outside `services/ai/`
- [ ] All DAGs/routers listed in §3.8 smoke-import cleanly

---

## 4. Open questions requiring confirmation before work continues

These aren't blockers for starting WO#12 (`recipe_agents.py`, which is
self-contained enough to proceed independently), but should be resolved
before the work order that needs each one specifically:

1. Which number is right — five or six remaining duplicate
   implementations (§1.5)? Affects `GOVERNANCE.md` §2.3's exact language.
2. Does `finance_upload.py` genuinely use the `google-genai` SDK, or was
   that WO#11 text imprecise (§1.5, §2 item 6)? Determines whether
   `services/ai/providers/gemini.py` needs SDK support at all.
3. Are `workout_plans.py` / `media_recommend.py`'s Gemini calls made from
   `async def` route handlers (§1.5)? Determines whether their migration
   needs a threadpool wrapper around `services/ai/base.py`'s blocking
   calls, or whether that's an existing, unrelated characteristic to
   leave alone.
4. Is there an AI-provider reference anywhere outside the six files
   tracked in §2 that hasn't surfaced in what's been reviewed so far
   (§3.7)? Only a full repo scan at WO-Final time will confirm.
