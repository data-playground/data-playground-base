"""add nba domain tables (teams, players, games, box score variants, play-by-play)

Revision ID: nb4_d0ma1n001
Revises: s0cc3r_d0ma1n001
Create Date: 2026-09-09

⚠ down_revision assumes s0cc3r_d0ma1n001 is your current single head. Before
applying this migration:
  1. Run `alembic heads`.
  2. If it prints exactly one hash and it's s0cc3r_d0ma1n001, you're good —
     apply as-is.
  3. If it prints a different single hash, change down_revision below to
     that hash.
  4. If it prints more than one, run
       alembic merge heads -m "merge before nba domain"
     first, then point down_revision at that new merge revision instead.

Creates the fourteen tables backing WO#33's NBA domain
(domains/nba/models.py):

  nba_teams               — the 30-franchise static reference table.
                             Seeded here with the same data
                             domains/nba/seed_data.py seeds idempotently
                             (ON DUPLICATE KEY UPDATE on tricode) — both are
                             safe to run, same reasoning as
                             soccer_competitions above: this migration
                             doesn't import seed_data.py, it duplicates the
                             list, so the migration has no dependency on
                             application code.

  nba_players              — synced weekly by
                             life_os_nba_player_sync.py. Not seeded here —
                             unlike nba_teams this is genuinely dynamic
                             data (~4,000+ current and historical players),
                             not a short static list worth hand-copying
                             into a migration. Starts empty; the first
                             player-sync run populates it.

  nba_games                — one row per game, upserted daily by
                             life_os_nba_ingest.py. Not seeded.

  nba_box_score_traditional,
  nba_box_score_advanced,
  nba_box_score_misc,
  nba_box_score_scoring,
  nba_box_score_usage,
  nba_box_score_fourfactors,
  nba_box_score_tracking,
  nba_box_score_hustle,
  nba_box_score_matchup,
  nba_box_score_defensive     — one row per player per game (matchup is
                             one row per defender/offensive-player pair
                             instead — see its own FK pair below). All ten
                             ingested by the same daily DAG as nba_games.
                             Not seeded. Only `traditional` has a dedicated
                             UI page in this pass — the other nine are
                             "headless" (browsable via /explorer) until the
                             agreed fast-follow builds first-class pages
                             for them; see models.py's module docstring.

  nba_play_by_play          — one row per action/event, upserted daily
                             alongside nba_games. Not seeded.

None of the eleven per-game tables get a bare single-column index on
game_id: each already declares a composite UNIQUE constraint whose
leftmost column is game_id (uq_bs_*_game_player, uq_bs_matchup_game_pair,
uq_pbp_game_action), and InnoDB can satisfy a game_id-only lookup from
that composite index's leftmost prefix — a separate index would be
redundant write overhead. nba_games.game_date gets its own index (below)
since nothing else covers date-only lookups.
"""
from typing import Sequence, Union
import datetime

from alembic import op
import sqlalchemy as sa

revision: str = 'nb4_d0ma1n001'
down_revision: Union[str, None] = 's0cc3r_d0ma1n001'  # ← verify per the note above
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Mirrors domains/nba/seed_data.py's TEAMS exactly. Duplicated here rather
# than imported — see module docstring.
SEED_TEAMS = [
    {"id": 1610612737, "tricode": "ATL", "full_name": "Atlanta Hawks", "conference": "East", "division": "Southeast"},
    {"id": 1610612738, "tricode": "BOS", "full_name": "Boston Celtics", "conference": "East", "division": "Atlantic"},
    {"id": 1610612751, "tricode": "BKN", "full_name": "Brooklyn Nets", "conference": "East", "division": "Atlantic"},
    {"id": 1610612766, "tricode": "CHA", "full_name": "Charlotte Hornets", "conference": "East", "division": "Southeast"},
    {"id": 1610612741, "tricode": "CHI", "full_name": "Chicago Bulls", "conference": "East", "division": "Central"},
    {"id": 1610612739, "tricode": "CLE", "full_name": "Cleveland Cavaliers", "conference": "East", "division": "Central"},
    {"id": 1610612742, "tricode": "DAL", "full_name": "Dallas Mavericks", "conference": "West", "division": "Southwest"},
    {"id": 1610612743, "tricode": "DEN", "full_name": "Denver Nuggets", "conference": "West", "division": "Northwest"},
    {"id": 1610612765, "tricode": "DET", "full_name": "Detroit Pistons", "conference": "East", "division": "Central"},
    {"id": 1610612744, "tricode": "GSW", "full_name": "Golden State Warriors", "conference": "West", "division": "Pacific"},
    {"id": 1610612745, "tricode": "HOU", "full_name": "Houston Rockets", "conference": "West", "division": "Southwest"},
    {"id": 1610612754, "tricode": "IND", "full_name": "Indiana Pacers", "conference": "East", "division": "Central"},
    {"id": 1610612746, "tricode": "LAC", "full_name": "LA Clippers", "conference": "West", "division": "Pacific"},
    {"id": 1610612747, "tricode": "LAL", "full_name": "Los Angeles Lakers", "conference": "West", "division": "Pacific"},
    {"id": 1610612763, "tricode": "MEM", "full_name": "Memphis Grizzlies", "conference": "West", "division": "Southwest"},
    {"id": 1610612748, "tricode": "MIA", "full_name": "Miami Heat", "conference": "East", "division": "Southeast"},
    {"id": 1610612749, "tricode": "MIL", "full_name": "Milwaukee Bucks", "conference": "East", "division": "Central"},
    {"id": 1610612750, "tricode": "MIN", "full_name": "Minnesota Timberwolves", "conference": "West", "division": "Northwest"},
    {"id": 1610612740, "tricode": "NOP", "full_name": "New Orleans Pelicans", "conference": "West", "division": "Southwest"},
    {"id": 1610612752, "tricode": "NYK", "full_name": "New York Knicks", "conference": "East", "division": "Atlantic"},
    {"id": 1610612760, "tricode": "OKC", "full_name": "Oklahoma City Thunder", "conference": "West", "division": "Northwest"},
    {"id": 1610612753, "tricode": "ORL", "full_name": "Orlando Magic", "conference": "East", "division": "Southeast"},
    {"id": 1610612755, "tricode": "PHI", "full_name": "Philadelphia 76ers", "conference": "East", "division": "Atlantic"},
    {"id": 1610612756, "tricode": "PHX", "full_name": "Phoenix Suns", "conference": "West", "division": "Pacific"},
    {"id": 1610612757, "tricode": "POR", "full_name": "Portland Trail Blazers", "conference": "West", "division": "Northwest"},
    {"id": 1610612758, "tricode": "SAC", "full_name": "Sacramento Kings", "conference": "West", "division": "Pacific"},
    {"id": 1610612759, "tricode": "SAS", "full_name": "San Antonio Spurs", "conference": "West", "division": "Southwest"},
    {"id": 1610612761, "tricode": "TOR", "full_name": "Toronto Raptors", "conference": "East", "division": "Atlantic"},
    {"id": 1610612762, "tricode": "UTA", "full_name": "Utah Jazz", "conference": "West", "division": "Northwest"},
    {"id": 1610612764, "tricode": "WAS", "full_name": "Washington Wizards", "conference": "East", "division": "Southeast"},
]

# One row per player-per-game box-score variant, keyed by target table name
# -> list of (column_name, column_type) pairs for its statistics block.
# Identity columns (id/game_id/team_id/person_id/position) are added by
# _create_player_stat_table() below for all of these — only the
# stat-specific columns differ per table, mirroring
# domains/nba/models.py's _PlayerGameStatMixin split.
_PLAYER_STAT_COLUMNS: dict[str, list[tuple[str, "sa.types.TypeEngine"]]] = {
    'nba_box_score_traditional': [
        ('minutes', sa.String(10)), ('field_goals_made', sa.Integer()),
        ('field_goals_attempted', sa.Integer()), ('field_goal_percentage', sa.Float()),
        ('three_pointers_made', sa.Integer()), ('three_pointers_attempted', sa.Integer()),
        ('three_pointer_percentage', sa.Float()), ('free_throws_made', sa.Integer()),
        ('free_throws_attempted', sa.Integer()), ('free_throw_percentage', sa.Float()),
        ('rebounds_offensive', sa.Integer()), ('rebounds_defensive', sa.Integer()),
        ('rebounds_total', sa.Integer()), ('assists', sa.Integer()), ('steals', sa.Integer()),
        ('blocks', sa.Integer()), ('turnovers', sa.Integer()), ('fouls_personal', sa.Integer()),
        ('points', sa.Integer()), ('plus_minus', sa.Float()),
    ],
    'nba_box_score_advanced': [
        ('minutes', sa.String(10)), ('estimated_offensive_rating', sa.Float()),
        ('offensive_rating', sa.Float()), ('estimated_defensive_rating', sa.Float()),
        ('defensive_rating', sa.Float()), ('estimated_net_rating', sa.Float()),
        ('net_rating', sa.Float()), ('assist_percentage', sa.Float()),
        ('assist_to_turnover', sa.Float()), ('assist_ratio', sa.Float()),
        ('offensive_rebound_percentage', sa.Float()), ('defensive_rebound_percentage', sa.Float()),
        ('rebound_percentage', sa.Float()), ('turnover_ratio', sa.Float()),
        ('effective_field_goal_percentage', sa.Float()), ('true_shooting_percentage', sa.Float()),
        ('usage_percentage', sa.Float()), ('estimated_usage_percentage', sa.Float()),
        ('estimated_pace', sa.Float()), ('pace', sa.Float()), ('pace_per_40', sa.Float()),
        ('possessions', sa.Float()), ('pie', sa.Float()),
    ],
    'nba_box_score_misc': [
        ('minutes', sa.String(10)), ('points_off_turnovers', sa.Integer()),
        ('points_second_chance', sa.Integer()), ('points_fast_break', sa.Integer()),
        ('points_paint', sa.Integer()), ('opp_points_off_turnovers', sa.Integer()),
        ('opp_points_second_chance', sa.Integer()), ('opp_points_fast_break', sa.Integer()),
        ('opp_points_paint', sa.Integer()), ('blocks', sa.Integer()), ('blocks_against', sa.Integer()),
        ('fouls_personal', sa.Integer()), ('fouls_drawn', sa.Integer()),
    ],
    'nba_box_score_scoring': [
        ('minutes', sa.String(10)), ('pct_field_goals_attempted_2pt', sa.Float()),
        ('pct_field_goals_attempted_3pt', sa.Float()), ('pct_points_2pt', sa.Float()),
        ('pct_points_midrange_2pt', sa.Float()), ('pct_points_3pt', sa.Float()),
        ('pct_points_fast_break', sa.Float()), ('pct_points_free_throw', sa.Float()),
        ('pct_points_off_turnovers', sa.Float()), ('pct_points_paint', sa.Float()),
        ('pct_assisted_2pt', sa.Float()), ('pct_unassisted_2pt', sa.Float()),
        ('pct_assisted_3pt', sa.Float()), ('pct_unassisted_3pt', sa.Float()),
        ('pct_assisted_fgm', sa.Float()), ('pct_unassisted_fgm', sa.Float()),
    ],
    'nba_box_score_usage': [
        ('minutes', sa.String(10)), ('usage_percentage', sa.Float()),
        ('pct_field_goals_made', sa.Float()), ('pct_field_goals_attempted', sa.Float()),
        ('pct_three_pointers_made', sa.Float()), ('pct_three_pointers_attempted', sa.Float()),
        ('pct_free_throws_made', sa.Float()), ('pct_free_throws_attempted', sa.Float()),
        ('pct_rebounds_offensive', sa.Float()), ('pct_rebounds_defensive', sa.Float()),
        ('pct_rebounds_total', sa.Float()), ('pct_assists', sa.Float()), ('pct_turnovers', sa.Float()),
        ('pct_steals', sa.Float()), ('pct_blocks', sa.Float()), ('pct_blocks_allowed', sa.Float()),
        ('pct_personal_fouls', sa.Float()), ('pct_personal_fouls_drawn', sa.Float()),
        ('pct_points', sa.Float()),
    ],
    'nba_box_score_fourfactors': [
        ('minutes', sa.String(10)), ('effective_field_goal_percentage', sa.Float()),
        ('free_throw_attempt_rate', sa.Float()), ('team_turnover_percentage', sa.Float()),
        ('offensive_rebound_percentage', sa.Float()), ('opp_effective_field_goal_percentage', sa.Float()),
        ('opp_free_throw_attempt_rate', sa.Float()), ('opp_team_turnover_percentage', sa.Float()),
        ('opp_offensive_rebound_percentage', sa.Float()),
    ],
    'nba_box_score_tracking': [
        ('minutes', sa.String(10)), ('distance', sa.Float()),
        ('rebound_chances_offensive', sa.Integer()), ('rebound_chances_defensive', sa.Integer()),
        ('rebound_chances_total', sa.Integer()), ('touches', sa.Integer()),
        ('secondary_assists', sa.Integer()), ('free_throw_assists', sa.Integer()),
        ('passes', sa.Integer()), ('assists', sa.Integer()),
        ('contested_field_goals_made', sa.Integer()), ('contested_field_goals_attempted', sa.Integer()),
        ('contested_field_goal_percentage', sa.Float()), ('uncontested_field_goals_made', sa.Integer()),
        ('uncontested_field_goals_attempted', sa.Integer()), ('uncontested_field_goal_percentage', sa.Float()),
        ('field_goal_percentage', sa.Float()), ('defended_at_rim_field_goals_made', sa.Integer()),
        ('defended_at_rim_field_goals_attempted', sa.Integer()),
        ('defended_at_rim_field_goal_percentage', sa.Float()),
    ],
    'nba_box_score_hustle': [
        ('minutes', sa.String(10)), ('points', sa.Integer()), ('contested_shots', sa.Integer()),
        ('contested_shots_2pt', sa.Integer()), ('contested_shots_3pt', sa.Integer()),
        ('deflections', sa.Integer()), ('charges_drawn', sa.Integer()),
        ('screen_assists', sa.Integer()), ('screen_assist_points', sa.Integer()),
        ('loose_balls_recovered_offensive', sa.Integer()), ('loose_balls_recovered_defensive', sa.Integer()),
        ('loose_balls_recovered_total', sa.Integer()), ('offensive_box_outs', sa.Integer()),
        ('defensive_box_outs', sa.Integer()), ('box_out_player_team_rebounds', sa.Integer()),
        ('box_out_player_rebounds', sa.Integer()), ('box_outs', sa.Integer()),
    ],
    'nba_box_score_defensive': [
        ('matchup_minutes', sa.String(10)), ('partial_possessions', sa.Float()),
        ('switches_on', sa.Integer()), ('player_points', sa.Integer()),
        ('defensive_rebounds', sa.Integer()), ('matchup_assists', sa.Integer()),
        ('matchup_turnovers', sa.Integer()), ('steals', sa.Integer()), ('blocks', sa.Integer()),
        ('matchup_field_goals_made', sa.Integer()), ('matchup_field_goals_attempted', sa.Integer()),
        ('matchup_field_goal_percentage', sa.Float()), ('matchup_three_pointers_made', sa.Integer()),
        ('matchup_three_pointers_attempted', sa.Integer()), ('matchup_three_pointer_percentage', sa.Float()),
    ],
}

# (table_name, unique_constraint_name) — box score variants where every
# row's identity is (game_id, person_id). Insertion order here is also the
# creation order below (no FK dependencies between them, so order doesn't
# matter functionally, but keeping it aligned with models.py's declaration
# order makes the two easy to diff against each other).
_PLAYER_STAT_TABLES = [
    ('nba_box_score_traditional', 'uq_bs_trad_game_player'),
    ('nba_box_score_advanced', 'uq_bs_adv_game_player'),
    ('nba_box_score_misc', 'uq_bs_misc_game_player'),
    ('nba_box_score_scoring', 'uq_bs_score_game_player'),
    ('nba_box_score_usage', 'uq_bs_usage_game_player'),
    ('nba_box_score_fourfactors', 'uq_bs_four_game_player'),
    ('nba_box_score_tracking', 'uq_bs_track_game_player'),
    ('nba_box_score_hustle', 'uq_bs_hustle_game_player'),
    ('nba_box_score_defensive', 'uq_bs_def_game_player'),
]


def _create_player_stat_table(table_name: str, unique_name: str) -> None:
    """
    Creates one "one row per player per game" box-score table: the shared
    identity columns (matching domains/nba/models.py's
    _PlayerGameStatMixin) plus that table's own statistics columns from
    _PLAYER_STAT_COLUMNS.
    """
    op.create_table(
        table_name,
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('game_id', sa.String(12), sa.ForeignKey('nba_games.game_id'), nullable=False),
        sa.Column('team_id', sa.Integer(), sa.ForeignKey('nba_teams.id'), nullable=True),
        sa.Column('person_id', sa.Integer(), sa.ForeignKey('nba_players.person_id'), nullable=True),
        sa.Column('position', sa.String(10), nullable=True),
        *[sa.Column(name, col_type, nullable=True) for name, col_type in _PLAYER_STAT_COLUMNS[table_name]],
    )
    op.create_unique_constraint(unique_name, table_name, ['game_id', 'person_id'])


def upgrade() -> None:
    # ── nba_teams ─────────────────────────────────────────────────────────
    op.create_table(
        'nba_teams',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=False),  # NBA's own team ID
        sa.Column('tricode', sa.String(3), nullable=False, unique=True),
        sa.Column('full_name', sa.String(60), nullable=False),
        sa.Column('conference', sa.String(10), nullable=True),
        sa.Column('division', sa.String(20), nullable=True),
    )
    op.bulk_insert(
        sa.table('nba_teams',
                 sa.column('id', sa.Integer),
                 sa.column('tricode', sa.String),
                 sa.column('full_name', sa.String),
                 sa.column('conference', sa.String),
                 sa.column('division', sa.String)),
        SEED_TEAMS,
    )

    # ── nba_players ───────────────────────────────────────────────────────
    op.create_table(
        'nba_players',
        sa.Column('person_id', sa.Integer(), primary_key=True, autoincrement=False),  # NBA's own player ID
        sa.Column('full_name', sa.String(120), nullable=False),
        sa.Column('team_id', sa.Integer(), sa.ForeignKey('nba_teams.id'), nullable=True),
        sa.Column('roster_status', sa.Integer(), nullable=True),
        sa.Column('from_year', sa.String(4), nullable=True),
        sa.Column('to_year', sa.String(4), nullable=True),
    )

    # ── nba_games ─────────────────────────────────────────────────────────
    op.create_table(
        'nba_games',
        sa.Column('game_id', sa.String(12), primary_key=True),  # NBA's own ID, e.g. "0022400231"
        sa.Column('season', sa.String(7), nullable=True),
        sa.Column('game_date', sa.Date(), nullable=False),
        sa.Column('game_status', sa.Integer(), nullable=True),
        sa.Column('game_status_text', sa.String(40), nullable=True),
        sa.Column('home_team_id', sa.Integer(), sa.ForeignKey('nba_teams.id'), nullable=True),
        sa.Column('away_team_id', sa.Integer(), sa.ForeignKey('nba_teams.id'), nullable=True),
        sa.Column('home_score', sa.Integer(), nullable=True),
        sa.Column('away_score', sa.Integer(), nullable=True),
        sa.Column('home_wins', sa.Integer(), nullable=True),
        sa.Column('home_losses', sa.Integer(), nullable=True),
        sa.Column('away_wins', sa.Integer(), nullable=True),
        sa.Column('away_losses', sa.Integer(), nullable=True),
        sa.Column('period', sa.Integer(), nullable=True),
        sa.Column('duration', sa.String(10), nullable=True),
        sa.Column('arena_name', sa.String(80), nullable=True),
        sa.Column('arena_city', sa.String(60), nullable=True),
        sa.Column('arena_state', sa.String(20), nullable=True),
        sa.Column('attendance', sa.Integer(), nullable=True),
        sa.Column('game_label', sa.String(60), nullable=True),
        sa.Column('game_sub_label', sa.String(60), nullable=True),
        sa.Column('series_text', sa.String(60), nullable=True),
    )
    op.create_index('ix_nba_games_game_date', 'nba_games', ['game_date'])

    # ── nine "one row per player per game" box-score variants ──────────────
    # traditional/advanced/misc/scoring/usage/fourfactors/tracking/hustle/
    # defensive — see _create_player_stat_table() and _PLAYER_STAT_COLUMNS
    # above for what actually varies between them.
    for table_name, unique_name in _PLAYER_STAT_TABLES:
        _create_player_stat_table(table_name, unique_name)

    # ── nba_box_score_matchup ────────────────────────────────────────────
    # One row per (defender, offensive player) pair, not one row per
    # player — doesn't fit _create_player_stat_table()'s shape.
    op.create_table(
        'nba_box_score_matchup',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('game_id', sa.String(12), sa.ForeignKey('nba_games.game_id'), nullable=False),
        sa.Column('team_id', sa.Integer(), sa.ForeignKey('nba_teams.id'), nullable=True),
        sa.Column('defender_person_id', sa.Integer(), sa.ForeignKey('nba_players.person_id'), nullable=False),
        sa.Column('defender_position', sa.String(10), nullable=True),
        sa.Column('offensive_person_id', sa.Integer(), sa.ForeignKey('nba_players.person_id'), nullable=False),
        sa.Column('matchup_minutes', sa.String(10), nullable=True),
        sa.Column('matchup_minutes_sort', sa.Float(), nullable=True),
        sa.Column('partial_possessions', sa.Float(), nullable=True),
        sa.Column('pct_defender_total_time', sa.Float(), nullable=True),
        sa.Column('pct_offensive_total_time', sa.Float(), nullable=True),
        sa.Column('pct_total_time_both_on', sa.Float(), nullable=True),
        sa.Column('switches_on', sa.Integer(), nullable=True),
        sa.Column('player_points', sa.Integer(), nullable=True),
        sa.Column('team_points', sa.Integer(), nullable=True),
        sa.Column('matchup_assists', sa.Integer(), nullable=True),
        sa.Column('matchup_potential_assists', sa.Integer(), nullable=True),
        sa.Column('matchup_turnovers', sa.Integer(), nullable=True),
        sa.Column('matchup_blocks', sa.Integer(), nullable=True),
        sa.Column('matchup_field_goals_made', sa.Integer(), nullable=True),
        sa.Column('matchup_field_goals_attempted', sa.Integer(), nullable=True),
        sa.Column('matchup_field_goals_percentage', sa.Float(), nullable=True),
        sa.Column('matchup_three_pointers_made', sa.Integer(), nullable=True),
        sa.Column('matchup_three_pointers_attempted', sa.Integer(), nullable=True),
        sa.Column('matchup_three_pointers_percentage', sa.Float(), nullable=True),
        sa.Column('help_blocks', sa.Integer(), nullable=True),
        sa.Column('help_field_goals_made', sa.Integer(), nullable=True),
        sa.Column('help_field_goals_attempted', sa.Integer(), nullable=True),
        sa.Column('help_field_goals_percentage', sa.Float(), nullable=True),
        sa.Column('matchup_free_throws_made', sa.Integer(), nullable=True),
        sa.Column('matchup_free_throws_attempted', sa.Integer(), nullable=True),
        sa.Column('shooting_fouls', sa.Integer(), nullable=True),
    )
    op.create_unique_constraint(
        'uq_bs_matchup_game_pair',
        'nba_box_score_matchup',
        ['game_id', 'defender_person_id', 'offensive_person_id'],
    )

    # ── nba_play_by_play ─────────────────────────────────────────────────
    op.create_table(
        'nba_play_by_play',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('game_id', sa.String(12), sa.ForeignKey('nba_games.game_id'), nullable=False),
        sa.Column('action_number', sa.Integer(), nullable=False),
        sa.Column('action_id', sa.Integer(), nullable=True),
        sa.Column('period', sa.Integer(), nullable=True),
        sa.Column('clock', sa.String(20), nullable=True),
        sa.Column('team_id', sa.Integer(), sa.ForeignKey('nba_teams.id'), nullable=True),
        sa.Column('team_tricode', sa.String(3), nullable=True),
        sa.Column('person_id', sa.Integer(), sa.ForeignKey('nba_players.person_id'), nullable=True),
        sa.Column('player_name', sa.String(80), nullable=True),
        sa.Column('action_type', sa.String(40), nullable=True),
        sa.Column('sub_type', sa.String(40), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('score_home', sa.Integer(), nullable=True),
        sa.Column('score_away', sa.Integer(), nullable=True),
        sa.Column('points_total', sa.Integer(), nullable=True),
        sa.Column('shot_distance', sa.Integer(), nullable=True),
        sa.Column('shot_result', sa.String(10), nullable=True),
        sa.Column('shot_value', sa.Integer(), nullable=True),
        sa.Column('is_field_goal', sa.Boolean(), nullable=True),
        sa.Column('location', sa.String(2), nullable=True),
        sa.Column('shot_x', sa.Integer(), nullable=True),
        sa.Column('shot_y', sa.Integer(), nullable=True),
    )
    op.create_unique_constraint(
        'uq_pbp_game_action', 'nba_play_by_play', ['game_id', 'action_number'],
    )


def downgrade() -> None:
    op.drop_table('nba_play_by_play')
    op.drop_table('nba_box_score_matchup')
    for table_name, _ in reversed(_PLAYER_STAT_TABLES):
        op.drop_table(table_name)
    op.drop_table('nba_games')
    op.drop_table('nba_players')
    op.drop_table('nba_teams')
