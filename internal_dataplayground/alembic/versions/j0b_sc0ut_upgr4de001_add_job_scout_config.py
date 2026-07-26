"""add job scout config tables and ATS source tracking

Revision ID: j0b_sc0ut_upgr4de001
Revises: w33kly_pl4nn3r001
Create Date: 2026-07-25

⚠ down_revision is a BEST GUESS based on the migration files you shared —
your alembic history looked like it had a few unmerged heads
(w0rk0ut_tr4ck3r001, m3d14_s34s0ns001, w33kly_pl4nn3r001 all branch without
a visible later merge). Run `alembic heads` first:
  - If it prints exactly one hash, replace down_revision below with it.
  - If it prints more than one, run
      alembic merge heads -m "merge before job scout config"
    first, then point down_revision at that new merge revision instead.

Creates:
  - job_search_keywords  (replaces the hardcoded DEFAULT_SEARCHES list in
    life_os_job_scout.py — the DAG now reads active keywords from here,
    editable from the Jobs > Config page instead of the codebase)
  - watched_companies    (the curated Greenhouse/Lever company list for the
    ATS fetchers in job_ats_agents.py — also editable from Jobs > Config)

Alters:
  - linkedin_jobs — adds `source` (default 'linkedin', so every existing
    row backfills correctly with zero manual data migration) and
    `external_ref` (nullable VARCHAR, backfilled from job_id for existing
    rows). external_ref exists because Lever's job IDs are UUID strings —
    they don't fit the existing job_id BigInteger column, and widening that
    column felt riskier than adding a parallel one. New uniqueness is
    enforced on (source, external_ref), not on job_id alone.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'j0b_sc0ut_upgr4de001'
down_revision: Union[str, None] = '1a04df9bdedb'  # ← verify per the note above
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Seeded from the DEFAULT_SEARCHES list already running in life_os_job_scout.py
# so nothing changes behaviorally on upgrade — you're just moving the same
# list into a table you can edit from the UI afterward.
SEED_KEYWORDS = [
    "Senior Analytics Engineer", "AI Solutions Architect", "Senior BI Engineer",
    "Senior Data Analyst", "Senior Data Scientist", "Senior Data Engineer",
    "Senior Machine Learning Engineer", "Senior AI Engineer", "Senior Analytics Manager",
    "Data Engineer (GCP), Solutions Engineer (Vertex AI), Analytics Architect",
    "Revenue Operations Engineer", "Product Data Scientist",
    "Senior Data Engineer AND (GCP OR BigQuery) AND Full-time",
    "(AI Architect OR Analytics Engineer) AND (Vertex OR GenAI) -Contract",
    "(Analytics Engineer OR Data Engineer) AND (Python AND SQL) AND Senior",
]


def upgrade() -> None:
    # ── job_search_keywords ────────────────────────────────────────────────
    op.create_table(
        'job_search_keywords',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('keyword', sa.String(255), nullable=False, unique=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('notes', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.bulk_insert(
        sa.table('job_search_keywords',
                 sa.column('keyword', sa.String),
                 sa.column('is_active', sa.Boolean)),
        [{'keyword': kw, 'is_active': True} for kw in SEED_KEYWORDS],
    )

    # ── watched_companies ──────────────────────────────────────────────────
    # Empty on creation — populated manually or via the "candidate companies"
    # promotion signal (see routers/job_config.py).
    op.create_table(
        'watched_companies',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('company_name', sa.String(255), nullable=False, unique=True),
        sa.Column('greenhouse_slug', sa.String(100), nullable=True),
        sa.Column('lever_slug', sa.String(100), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        # Free-text breadcrumb for why this company was added, e.g.
        # "3 postings, avg fit 91 — promoted from LinkedIn scrape"
        sa.Column('source_note', sa.String(255), nullable=True),
        sa.Column('added_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # ── linkedin_jobs: source + external_ref ───────────────────────────────
    op.add_column(
        'linkedin_jobs',
        sa.Column('source', sa.String(20), nullable=False, server_default='linkedin'),
    )
    op.add_column(
        'linkedin_jobs',
        sa.Column('external_ref', sa.String(64), nullable=True),
    )

    # Backfill external_ref for all existing (LinkedIn) rows from job_id,
    # so every row — old and new — can be uniquely addressed by (source, external_ref).
    op.execute(
        "UPDATE linkedin_jobs SET external_ref = CAST(job_id AS CHAR) "
        "WHERE external_ref IS NULL AND job_id IS NOT NULL"
    )

    op.create_index('ix_linkedin_jobs_source', 'linkedin_jobs', ['source'])
    op.create_unique_constraint(
        'uq_linkedin_jobs_source_external_ref',
        'linkedin_jobs',
        ['source', 'external_ref'],
    )


def downgrade() -> None:
    op.drop_constraint('uq_linkedin_jobs_source_external_ref', 'linkedin_jobs', type_='unique')
    op.drop_index('ix_linkedin_jobs_source', table_name='linkedin_jobs')
    op.drop_column('linkedin_jobs', 'external_ref')
    op.drop_column('linkedin_jobs', 'source')

    op.drop_table('watched_companies')
    op.drop_table('job_search_keywords')
