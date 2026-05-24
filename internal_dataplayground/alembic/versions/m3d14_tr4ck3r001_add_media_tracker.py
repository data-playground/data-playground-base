"""add media tracker tables

Revision ID: m3d14_tr4ck3r001
Revises: r3c1p3_m4n4g3r001
Create Date: 2026-05-21

Creates:
  - media_items            (reference catalog: movies, TV shows, books)
  - user_media             (user's personal tracking records)
  - tv_season_progress     (per-season episode tracking for TV shows)
  - media_recommendations  (cached recommendation results)
  - streaming_services     (user's subscribed streaming platforms)

Design decisions:
  - embedding stored as JSON array (384-dim float vector from all-MiniLM-L6-v2)
    MariaDB has no native vector type; JSON is the cleanest option and avoids
    a dependency on pgvector or similar. Performance is acceptable at library sizes
    typical of a personal collection (<10K items).
  - user_rating stored 1-10 (half-stars = odd numbers) but displayed as stars in UI.
    1=½★  2=★  3=1½★  4=★★  5=2½★  6=★★★  7=3½★  8=★★★★  9=4½★  10=★★★★★
  - streaming_services table stores what the user subscribes to.
    The router uses is_subscribed to default recommendations to owned services,
    with an opt-in toggle to include non-subscribed services.
  - media_recommendations JSON schema: [{"media_item_id": int, "score": float,
    "title": str, "reasoning": str | null}]
  - UNIQUE KEY on (external_id, external_source) allows the same movie to exist
    once from TMDB and not be duplicated if searched twice.
  - tv_season_progress is sparse — only seasons the user has started appear here.
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = 'm3d14_tr4ck3r001'
down_revision: Union[str, None] = 'r3c1p3_m4n4g3r001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:

    # ── streaming_services ─────────────────────────────────────────────────────
    # Stores the user's streaming subscriptions.
    # is_subscribed=True means this service is active and preferred for
    # recommendations. The user can still request recommendations that include
    # non-subscribed services via the include_unsubscribed toggle.
    op.create_table(
        'streaming_services',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(100), nullable=False, unique=True),
        # Slug used to match TMDB provider names, e.g. "netflix", "prime"
        sa.Column('tmdb_provider_name', sa.String(100), nullable=True),
        # TMDB provider ID for direct API matching
        sa.Column('tmdb_provider_id', sa.Integer(), nullable=True),
        sa.Column('logo_url', sa.String(500), nullable=True),
        sa.Column('is_subscribed', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('sort_order', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_streaming_services_subscribed', 'streaming_services', ['is_subscribed'])

    # Seed common streaming services (user activates subscriptions in settings)
    op.bulk_insert(
        sa.table('streaming_services',
            sa.column('name', sa.String),
            sa.column('tmdb_provider_name', sa.String),
            sa.column('tmdb_provider_id', sa.Integer),
            sa.column('is_subscribed', sa.Boolean),
            sa.column('sort_order', sa.Integer),
        ),
        [
            {'name': 'Netflix',          'tmdb_provider_name': 'Netflix',           'tmdb_provider_id': 8,    'is_subscribed': False, 'sort_order': 1},
            {'name': 'Amazon Prime',     'tmdb_provider_name': 'Amazon Prime Video', 'tmdb_provider_id': 119,  'is_subscribed': False, 'sort_order': 2},
            {'name': 'Disney+',          'tmdb_provider_name': 'Disney Plus',        'tmdb_provider_id': 337,  'is_subscribed': False, 'sort_order': 3},
            {'name': 'Max',              'tmdb_provider_name': 'Max',                'tmdb_provider_id': 1899, 'is_subscribed': False, 'sort_order': 4},
            {'name': 'Hulu',             'tmdb_provider_name': 'Hulu',               'tmdb_provider_id': 15,   'is_subscribed': False, 'sort_order': 5},
            {'name': 'Apple TV+',        'tmdb_provider_name': 'Apple TV Plus',      'tmdb_provider_id': 350,  'is_subscribed': False, 'sort_order': 6},
            {'name': 'Paramount+',       'tmdb_provider_name': 'Paramount Plus',     'tmdb_provider_id': 531,  'is_subscribed': False, 'sort_order': 7},
            {'name': 'Peacock',          'tmdb_provider_name': 'Peacock',            'tmdb_provider_id': 386,  'is_subscribed': False, 'sort_order': 8},
            {'name': 'Crunchyroll',      'tmdb_provider_name': 'Crunchyroll',        'tmdb_provider_id': 283,  'is_subscribed': False, 'sort_order': 9},
            {'name': 'Tubi',             'tmdb_provider_name': 'Tubi TV',            'tmdb_provider_id': 73,   'is_subscribed': False, 'sort_order': 10},
        ]
    )

    # ── media_items ────────────────────────────────────────────────────────────
    # The catalog — one row per unique movie/show/book regardless of how many
    # times it was searched or from which source. The UNIQUE constraint on
    # (external_id, external_source) prevents duplication.
    # embedding is a 384-dim float array stored as JSON, generated by the
    # ml-service container running all-MiniLM-L6-v2.
    op.create_table(
        'media_items',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),

        # ── External source identity ───────────────────────────────────────────
        sa.Column('external_id', sa.String(50), nullable=True),
        sa.Column('external_source',
                  sa.Enum('tmdb_movie', 'tmdb_tv', 'openlibrary', 'manual',
                          name='mediaexternalsource'),
                  nullable=False, server_default='manual'),

        # ── Core metadata ──────────────────────────────────────────────────────
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('media_type',
                  sa.Enum('movie', 'tv_show', 'book', name='mediatype'),
                  nullable=False),
        sa.Column('genres', sa.JSON(), nullable=True),
        sa.Column('release_year', sa.SmallInteger(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('poster_url', sa.String(500), nullable=True),
        sa.Column('external_rating', sa.Numeric(3, 1), nullable=True),

        # ── Movie-specific ─────────────────────────────────────────────────────
        sa.Column('runtime_minutes', sa.Integer(), nullable=True),

        # ── TV-specific ────────────────────────────────────────────────────────
        sa.Column('total_seasons', sa.SmallInteger(), nullable=True),
        sa.Column('total_episodes', sa.SmallInteger(), nullable=True),

        # ── Book-specific ──────────────────────────────────────────────────────
        sa.Column('author', sa.String(255), nullable=True),
        sa.Column('page_count', sa.SmallInteger(), nullable=True),

        # ── Streaming availability ─────────────────────────────────────────────
        # JSON array of TMDB provider IDs available for streaming in the US.
        # Populated by the TMDB /watch/providers endpoint.
        # Example: [8, 119] means Netflix + Amazon Prime.
        sa.Column('streaming_provider_ids', sa.JSON(), nullable=True),
        sa.Column('streaming_fetched_at', sa.DateTime(), nullable=True),

        # ── ML embedding ──────────────────────────────────────────────────────
        # 384-dimension float vector from all-MiniLM-L6-v2.
        # Stored as JSON array; cosine similarity computed in Python.
        sa.Column('embedding', sa.JSON(), nullable=True),
        sa.Column('embedding_generated_at', sa.DateTime(), nullable=True),

        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    # UNIQUE on (external_id, external_source) — prevents duplicate TMDB entries
    op.create_unique_constraint(
        'uq_media_external',
        'media_items',
        ['external_id', 'external_source'],
    )
    op.create_index('ix_media_items_type', 'media_items', ['media_type'])
    op.create_index('ix_media_items_title', 'media_items', ['title'])

    # ── user_media ─────────────────────────────────────────────────────────────
    # The user's personal tracking record for each media item.
    # One row per media_item — the UNIQUE constraint prevents double-tracking.
    # mood_tags JSON stores both predefined and custom tags as a string array.
    op.create_table(
        'user_media',
        sa.Column('id', sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column('media_item_id', sa.Integer(),
                  sa.ForeignKey('media_items.id', ondelete='CASCADE'),
                  nullable=False, index=True),

        # ── Tracking state ─────────────────────────────────────────────────────
        sa.Column('status',
                  sa.Enum('want_to', 'in_progress', 'completed', 'abandoned',
                          name='usermediastatus'),
                  nullable=False, server_default='want_to'),

        # ── Personal rating: 1-10, displayed as half-stars (odd = half star) ───
        # 1=½★  2=★  3=1½★  4=★★  5=2½★  6=★★★  7=3½★  8=★★★★  9=4½★  10=★★★★★
        sa.Column('user_rating', sa.SmallInteger(), nullable=True),

        # ── Timeline ──────────────────────────────────────────────────────────
        sa.Column('started_at', sa.Date(), nullable=True),
        sa.Column('completed_at', sa.Date(), nullable=True),

        # ── Personal context ──────────────────────────────────────────────────
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('rewatch_count', sa.SmallInteger(), nullable=False, server_default='0'),

        # ── Mood tags: predefined + custom, stored as string JSON array ────────
        # Example: ["light", "cerebral", "my-friday-night-vibe"]
        sa.Column('mood_tags', sa.JSON(), nullable=True),

        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_unique_constraint('uq_user_media_item', 'user_media', ['media_item_id'])
    op.create_index('ix_user_media_status', 'user_media', ['status'])
    op.create_index('ix_user_media_rating', 'user_media', ['user_rating'])

    # ── tv_season_progress ─────────────────────────────────────────────────────
    # Sparse — only seasons the user has started appear here.
    # A TV show with 5 seasons where the user has watched seasons 1-3 will
    # have 3 rows here, not 5.
    op.create_table(
        'tv_season_progress',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('user_media_id', sa.BigInteger(),
                  sa.ForeignKey('user_media.id', ondelete='CASCADE'),
                  nullable=False, index=True),
        sa.Column('season_number', sa.SmallInteger(), nullable=False),
        sa.Column('episodes_watched', sa.SmallInteger(), nullable=False, server_default='0'),
        sa.Column('total_episodes', sa.SmallInteger(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_unique_constraint(
        'uq_season_progress', 'tv_season_progress', ['user_media_id', 'season_number']
    )

    # ── media_recommendations ──────────────────────────────────────────────────
    # Caches recommendation results to avoid re-running the ML pipeline on every
    # page load. Each row represents one recommendation session.
    # recommendations JSON schema:
    #   [{"media_item_id": int, "score": float, "title": str, "reasoning": str|null}]
    op.create_table(
        'media_recommendations',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('generated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('input_mood', sa.String(255), nullable=True),
        sa.Column('input_context', sa.String(255), nullable=True),
        sa.Column('media_type_filter',
                  sa.Enum('movie', 'tv_show', 'book', 'any', name='recommendationmediatype'),
                  nullable=False, server_default='any'),
        sa.Column('include_unsubscribed', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('used_gemini', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('recommendations', sa.JSON(), nullable=False),
        sa.Column('ml_model_version', sa.String(50), nullable=True),
    )
    op.create_index('ix_media_recommendations_generated', 'media_recommendations', ['generated_at'])


def downgrade() -> None:
    op.drop_index('ix_media_recommendations_generated', table_name='media_recommendations')
    op.drop_table('media_recommendations')

    op.drop_constraint('uq_season_progress', 'tv_season_progress', type_='unique')
    op.drop_table('tv_season_progress')

    op.drop_index('ix_user_media_rating', table_name='user_media')
    op.drop_index('ix_user_media_status', table_name='user_media')
    op.drop_constraint('uq_user_media_item', 'user_media', type_='unique')
    op.drop_table('user_media')

    op.drop_index('ix_media_items_title', table_name='media_items')
    op.drop_index('ix_media_items_type', table_name='media_items')
    op.drop_constraint('uq_media_external', 'media_items', type_='unique')
    op.drop_table('media_items')

    op.drop_index('ix_streaming_services_subscribed', table_name='streaming_services')
    op.drop_table('streaming_services')

    op.execute("DROP TYPE IF EXISTS recommendationmediatype")
    op.execute("DROP TYPE IF EXISTS usermediastatus")
    op.execute("DROP TYPE IF EXISTS mediatype")
    op.execute("DROP TYPE IF EXISTS mediaexternalsource")
