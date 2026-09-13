# New Work Orders — Index (WO#21–#35)

*Companion to `00_MASTER_INDEX.md`. Covers everything drafted for Tracks
B, C, D, and E — Track A is excluded per instruction, since it's already
mid-execution with WO#16 pending.*

| # | Track | Title | File | Status |
|---|---|---|---|---|
| 21 | B | GOVERNANCE.md §2.5 amendment | `work_order_21_governance_dag_amendment.md` | 📝 Drafted, ready to run |
| 22 | B | `models.py` end-state + DAG header cleanup + `configure_mappers()` | `work_order_22_models_dag_headers_configure_mappers.md` | 📝 Drafted, ready to run |
| 23 | C | Habits router split | `work_order_23_habits_router_split.md` | 📝 Drafted, ready to run |
| 24 | C | Code Intel router split (3 files) | `work_order_24_code_intel_router_split.md` | 📝 Drafted, ready to run |
| 25 | C | Journal router split | `work_order_25_journal_router_split.md` | 📝 Drafted, ready to run |
| 26 | C | Recipes router split (2 files) | `work_order_26_recipes_router_split.md` | 📝 Drafted, ready to run |
| 27 | C | Blog router split | `work_order_27_blog_router_split.md` | 📝 Drafted, ready to run |
| 28 | C | Media router split (2 files) | `work_order_28_media_router_split.md` | 📝 Drafted, ready to run |
| 29 | C | Workout router split (3 files, one blocked on sign-off) | `work_order_29_workout_router_split.md` | 📝 Drafted, ready to run (Part C needs owner sign-off before execution) |
| 30 | C | Planning router — verification only | `work_order_30_planning_router_verification.md` | 📝 Drafted, ready to run (likely a no-op) |
| — | D | Pre-flight risk analysis (not a formal WO) | `track_D_agent_reorg_scoping_analysis.md` | ✅ Complete — read before WO#31 |
| 31 | D | Tier 1 agent-module reorg (DAG-only) | `work_order_31_agent_reorg_tier1.md` | 📝 Drafted, ready to run |
| 32 | E | Blog Scout prompt/interest-alignment review | `work_order_32_blog_scout_prompt_review.md` | 📝 Drafted — waiting on the JSON export you're providing |
| 33 | E | NBA data domain (new build) | `work_order_33_nba_domain.md` | 📝 Drafted — Step 1 is a materials request, not executable yet |
| 34 | E | Soccer data domain (new build) | `work_order_34_soccer_domain.md` | 📝 Drafted — Step 1 is a materials request, not executable yet |
| 35 | E | Medium article extraction (revisit) | `work_order_35_medium_domain_revisit.md` | 📝 Drafted — Step 1 is a materials request, not executable yet |

**Execution notes:**
- WO#23–30 (Track C) and WO#31 (Track D) can all run in parallel with each
  other and with the ongoing Track A work — none share a file.
- WO#21 and WO#22 (Track B) are independent of everything else and of each
  other.
- WO#29's Part C (`workout_log.py`) is the one item in this whole batch that
  is explicitly **not** authorized to execute without a separate owner
  sign-off — see that file's Part C.
- WO#33–35 cannot really start until their Step 1 materials requests are
  answered — they're drafted so that whichever agent picks them up knows
  exactly what to ask for and what shape the eventual build should take,
  without being told *how* to build it (per your instruction — these three
  are guided, not restricted).
- WO#32 is ready to run the moment the JSON export lands.

**Superseded:** earlier in this conversation I produced three bundled
drafts (`track_B_loose_ends_work_orders.md`,
`track_C_router_line_limit_work_orders.md`,
`track_D_agent_folder_reorganization_scoping.md`,
`track_E_status_and_blog_scout_work_order.md`) before switching to
one-file-per-work-order. Those are superseded by the files in this index —
safe to discard.
