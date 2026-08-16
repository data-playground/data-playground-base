# Work Order #19 — Dead Code Removal + Router Line-Limit Lint Script

*Two small, independent, low-risk cleanup items bundled together because
neither is large enough to justify its own full work order, and neither
depends on any other work order in this series having been executed first.
Both can run immediately.*

---

## ROLE
You are a senior refactoring engineer performing two small, independent
cleanup tasks. Keep them separate in your diff and report — do not let
either task's changes bleed into the other's files.

## HARD BOUNDARIES
- Only read/edit files explicitly listed in SCOPE below.
- **Task 1 (dead code removal) touches ONLY the specific commented-out
  block identified below in `blog_agents.py` — nothing else in that file
  changes.** This is the block that predates every work order in this
  series (it's present in the codebase as given, not something introduced
  by WO#11–16) — it is the old, superseded `_cerebras()` implementation
  that was already fully commented out before this engagement began.
- **Task 2 (lint script) creates a new, standalone script — it does not
  modify any existing router file, does not wire itself into any CI
  configuration file** (none was provided in this codebase's file list to
  wire into — inventing one would be guessing at infrastructure that may
  not exist in the form assumed). The script should be runnable manually
  (`python scripts/check_router_line_limits.py`) and exit non-zero on
  violation, so it's ready to be wired into whatever CI system is actually
  in use, without this work order guessing what that is.

## HANDLING PRE-EXISTING BUGS DISCOVERED DURING VERIFICATION
If, while verifying an acceptance criterion, you discover the app behaves
incorrectly in a way that is unrelated to this work, do not fix it —
report it under Notes per the standard rule used throughout this series.

## WORKING METHOD
Complete Task 1 fully (including verification) before starting Task 2 —
they're independent, but keeping them sequential keeps the diff easy to
review in two clean pieces.

## OUTPUT FORMAT
1. **Files created**
2. **Files moved**
3. **Files edited** (path — description)
4. **Acceptance criteria results** (✅/❌/⚠️ + one-line reason for non-✅)
5. **Notes**

## ROLLBACK
`git checkout` on every file listed in sections 1–3 of the output above.

---

## SCOPE

**Task 1 — file to edit:**
- `airflow/agents/blog_agents.py`

**Task 2 — new file to create:**
- `scripts/check_router_line_limits.py`

**Task 2 — files to read (not edit) to determine current violations:**
- Every `.py` file under `routers/` (root, for domains not yet migrated)
  and every `.py` file under `domains/*/routers/` (for domains already
  migrated, if any have run by the time this executes)

---

## STEPS

### Task 1 — Dead Code Removal

1. Locate the fully commented-out prior `_cerebras()` implementation in
   `blog_agents.py` — it appears directly above the live, working
   `_cerebras()` function, is introduced by a commented-out `# def
   _cerebras(` line, and every line within it (including its docstring,
   its retry loop, its exception handling) is prefixed with `#`. It ends
   just before the `# ── CEREBRAS MODEL IDs ──` section header.

2. Delete this entire commented block. Do not touch the live
   `_cerebras()` function that follows it, and do not touch anything
   above the dead block (the earlier commented-out prior attempt shown
   with `# INTER_REQUEST_DELAY_SEC` context, if present as a separate
   dead fragment — confirm whether that's a second, distinct piece of
   dead code or part of the same block, and report which you found; only
   remove what you can clearly confirm is dead/superseded, leave anything
   ambiguous in place and flag it under Notes instead of guessing).

3. Confirm the file still parses correctly and every live function
   (`_cerebras`, `agent_code_narrator`, `agent_refiner`,
   `agent_code_commenter`, `agent_code_improver`, and every other function
   in the file) is textually unchanged aside from the deleted block.

### Task 2 — Router Line-Limit Lint Script

4. Create `scripts/check_router_line_limits.py`:
   ```python
   #!/usr/bin/env python3
   """
   Enforces GOVERNANCE.md §1.2: routers must not exceed 300 lines.

   Usage:
       python scripts/check_router_line_limits.py

   Exits 0 if every router file is within the limit, exits 1 and prints
   every violation (path + line count) otherwise. Intended to be wired
   into whatever CI system this project uses — this script itself makes
   no assumption about which one that is.
   """
   import sys
   from pathlib import Path

   LINE_LIMIT = 300
   SEARCH_ROOTS = ["routers", "domains"]  # domains/*/routers/ included via glob below

   def find_router_files(repo_root: Path) -> list[Path]:
       """
       Returns every .py file under routers/ (root-level, for domains not
       yet migrated into domains/) and every .py file under any
       domains/*/routers/ directory.
       """
       # implementation: glob repo_root/"routers"/**/*.py
       # plus repo_root/"domains"/*/"routers"/**/*.py

   def main() -> int:
       # walk found files, count lines, report violations, return exit code
       ...

   if __name__ == "__main__":
       sys.exit(main())
   ```
   Fill in the implementation per the docstring's contract. Exclude
   `__init__.py` files from the line count (they're expected to be small
   or empty and aren't meaningfully "routers" in the sense the 300-line
   rule targets).

5. Run the script against the current repository state and report its
   output as-is in your report — do not editorialize on the results or
   attempt to fix any violation it finds (that's a separate, later task
   per GOVERNANCE.md §1.2's own note that oversized routers get split
   "before they cross 300, not after," which is a design task, not
   something this lint script itself should attempt).

---

## ACCEPTANCE CRITERIA

- [ ] The dead `_cerebras()` block no longer exists in `blog_agents.py` —
  confirm via `grep` that no fully-commented duplicate of the retry logic
  remains
- [ ] Every live function in `blog_agents.py` is textually unchanged
  except for the deletion — confirm via `git diff` showing only the dead
  block's lines removed, nothing else touched
- [ ] `scripts/check_router_line_limits.py` runs successfully
  (`python scripts/check_router_line_limits.py`) and exits with the
  correct code (0 or 1) matching whether any violation actually exists in
  the current repo state
- [ ] The script correctly finds router files in **both** possible
  locations (`routers/*.py` for not-yet-migrated domains and
  `domains/*/routers/*.py` for migrated ones) — since this work order may
  run before or after any of WO#2–10, the script must work correctly
  regardless of how many domains have actually been migrated at the time
  it's run; verify this by confirming it finds the expected router files
  given the current state of the repo, whatever that state is
- [ ] The script's reported violations (if any) are included verbatim in
  your report, unmodified and unfixed

---

## For the next work order (not part of this one)

**Work Order #20** should be the shim-removal pass (GOVERNANCE.md §2.4) —
but it is explicitly **conditional on WO#2–10 having actually been
executed first**, since it can only meaningfully check "is `dashboard.py`
still the only consumer of this domain's shim" for domains that have
actually been migrated. That work order should be written as a
per-domain checklist that's safe to run partially (skip any domain not
yet migrated, note it as "not applicable yet" rather than treating it as
a failure) — draft it whenever you're ready to actually execute WO#2–10
and want the cleanup ready to follow immediately after, or hold off
drafting it until then, since its exact shape may depend on what WO#2–10's
execution reports actually reveal.
