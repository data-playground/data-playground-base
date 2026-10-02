"""add explorer_table_domains (table -> domain mapping for the SQL Explorer)

Revision ID: expl0r3r_d0ma1ns001
Revises: s0cc3r_st4ts001
Create Date: 2026-09-21

Creates explorer_table_domains and seeds it with the current table -> domain
assignments (69 existing tables, plus this new table itself under "explorer").

⚠ down_revision was written against s0cc3r_st4ts001, the head at the time of
writing. Per this project's established pattern of the head moving between
sessions, don't take that on faith:
  1. Run `alembic heads`.
  2. If it prints exactly one hash and it's s0cc3r_st4ts001, apply as-is.
  3. If it prints a different single hash, change down_revision below to
     that hash.
  4. If it prints more than one, run
       alembic merge heads -m "merge before explorer domains"
     first, then point down_revision at that new merge revision instead.

Tables created after this migration need nothing here: they show up in the
explorer under "unassigned" until assigned on /explorer/settings.

The seed is only a starting point — the settings page is the source of truth
afterwards, so downgrade() drops the whole table (and any edits made there).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'expl0r3r_d0ma1ns001'
down_revision: Union[str, None] = 's0cc3r_st4ts001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_SEED = {
    "blog": [
        "blog_ideas",
    ],
    "code_intel": [
        "code_files",
        "code_projects",
        "folder_readmes",
    ],
    "explorer": [
        "explorer_table_domains",
    ],
    "finance": [
        "accounts",
        "categories",
        "transactions",
    ],
    "habits": [
        "habit_logs",
        "habit_settings",
        "habits",
    ],
    "jobs": [
        "application_logs",
        "job_scout_run_log",
        "job_search_keywords",
        "linkedin_jobs",
        "staging_jobs",
        "watched_companies",
    ],
    "journal": [
        "journal_entries",
        "weekly_syntheses",
    ],
    "life_os": [
        "alembic_version",
    ],
    "media": [
        "media_items",
        "media_recommendations",
        "streaming_services",
        "tv_season_progress",
        "user_media",
    ],
    "medium": [
        "medium_articles",
        "medium_feed_sources",
    ],
    "nba": [
        "nba_box_score_advanced",
        "nba_box_score_defensive",
        "nba_box_score_fourfactors",
        "nba_box_score_hustle",
        "nba_box_score_matchup",
        "nba_box_score_misc",
        "nba_box_score_scoring",
        "nba_box_score_tracking",
        "nba_box_score_traditional",
        "nba_box_score_usage",
        "nba_games",
        "nba_play_by_play",
        "nba_players",
        "nba_teams",
    ],
    "planning": [
        "shopping_lists",
        "user_intent",
        "weekly_plan_days",
        "weekly_plan_meals",
        "weekly_plans",
    ],
    "recipes": [
        "ingredients",
        "pantry_items",
        "recipe_ingredients",
        "recipe_tags",
        "recipe_tags_junction",
        "recipes",
    ],
    "soccer": [
        "soccer_bookings",
        "soccer_coaches",
        "soccer_competitions",
        "soccer_goals",
        "soccer_match_lineups",
        "soccer_matches",
        "soccer_raw_payloads",
        "soccer_settings",
        "soccer_substitutions",
    ],
    "workout": [
        "body_metrics",
        "equipment",
        "exercises",
        "workout_locations",
        "workout_plan_days",
        "workout_plan_exercises",
        "workout_plans",
        "workout_sessions",
        "workout_sets",
    ],
}


def upgrade() -> None:
    t = op.create_table(
        "explorer_table_domains",
        sa.Column("table_name", sa.String(64), primary_key=True),
        sa.Column("domain", sa.String(50), nullable=False),
    )
    op.create_index(
        "ix_explorer_table_domains_domain", "explorer_table_domains", ["domain"]
    )
    op.bulk_insert(t, [
        {"table_name": tbl, "domain": dom}
        for dom, tbls in _SEED.items() for tbl in tbls
    ])


def downgrade() -> None:
    op.drop_index("ix_explorer_table_domains_domain", table_name="explorer_table_domains")
    op.drop_table("explorer_table_domains")
