# domains/nba/models.py
"""
NBA domain — ORM models (WO#33).

Schema shape: normalized relational tables, not a 1:1 port of the original
BigQuery nested-RECORD schema in nba_game_summary.py. MariaDB has no
equivalent of BigQuery's RECORD/REPEATED column types, so every
per-player-per-game statistics block from the source script becomes its
own flat table — one row per player per game — joined back to nba_games /
nba_teams / nba_players by foreign key. This was the one genuinely
hard-to-reverse call in this build and was confirmed with the project
owner before writing it (see WO#33's Step 1/Step 2 exchange).

v1 dedicated UI (domains/nba/routers/nba_games.py) covers only Game,
BoxScoreTraditional, and PlayByPlayEvent. The other nine box-score variant
tables below (Advanced, Misc, Scoring, Usage, FourFactors, Tracking,
Hustle, Matchup, Defensive) are ingested by the DAG on the same schedule
but have no dedicated page yet in this pass — they're intentionally
"headless," browsable via the existing /explorer (SQL Explorer) domain.
Building first-class UI for them is the agreed fast-follow.

Column sets are ported directly from each endpoint's `fields` skeleton in
nba_game_summary.py's VAR_NAMING_GUIDE — see that file for the original
field list each table below is normalizing out of a nested JSON blob.

Every table here is written to exclusively by the ingestion DAGs under
airflow/dags/nba/, via raw SQL (per CONTRIBUTING.md / GOVERNANCE.md §2.2 —
DAGs never import models.py). This file exists for the FastAPI routers to
query, and as the source of truth for the Alembic migration that creates
these tables.
"""
from sqlalchemy import (
    Boolean, Column, Date, Float, ForeignKey, Integer, String, Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from core.base_model import Base


# ── Reference / dimension tables ────────────────────────────────────────────

class Team(Base):
    """
    Static reference table — the 30 NBA franchises. Seeded once via
    seed_data.py (run manually after migrating), not written to by the
    ingestion DAG. Team relocations/rebrands are rare enough that a static
    seed is simpler and more reliable than deriving team metadata from
    box-score responses, which only ever carry a bare teamId — never a
    name or tricode.
    """
    __tablename__ = "nba_teams"

    id = Column(Integer, primary_key=True, autoincrement=False)  # NBA's own numeric team ID
    tricode = Column(String(3), nullable=False, unique=True)
    full_name = Column(String(60), nullable=False)
    conference = Column(String(10), nullable=True)   # "East" / "West"
    division = Column(String(20), nullable=True)


class Player(Base):
    """
    Synced weekly from the PLAYERS endpoint (commonallplayers) by
    life_os_nba_player_sync.py — a full roster snapshot, not per-game data,
    so a weekly cadence is enough. Exists so the box score / play-by-play
    UI can show real names instead of bare person IDs.

    Full biographical detail (birthdate, college, draft info — the
    PLAYER_DETAIL endpoint in the original script) is deferred to the
    fast-follow pass, same as the other nine box-score variants.
    """
    __tablename__ = "nba_players"

    person_id = Column(Integer, primary_key=True, autoincrement=False)  # NBA's own numeric player ID
    full_name = Column(String(120), nullable=False)
    team_id = Column(Integer, ForeignKey("nba_teams.id"), nullable=True)
    roster_status = Column(Integer, nullable=True)   # NBA's own convention: 1 = active, 0 = inactive
    from_year = Column(String(4), nullable=True)
    to_year = Column(String(4), nullable=True)


# ── Games ────────────────────────────────────────────────────────────────────

class Game(Base):
    """
    One row per game. Populated in two passes by the ingestion DAG:
      1. Discovery (GAME_DATA/gamecardfeed, by date) — game_id, game_date,
         game_status, game_status_text.
      2. Box-score summary (BS_SUMMARY) — fills in scores, team IDs, arena,
         attendance, and everything else, once the game has started.
    A row can legitimately exist with only discovery fields populated (a
    game later tonight that hasn't tipped off yet) — every non-key column
    is nullable for exactly this reason.
    """
    __tablename__ = "nba_games"

    game_id = Column(String(12), primary_key=True)    # NBA's own ID, e.g. "0022400231"
    season = Column(String(7), nullable=True)          # derived, e.g. "2024-25"
    game_date = Column(Date, nullable=False, index=True)

    game_status = Column(Integer, nullable=True)       # 1=scheduled, 2=live, 3=final
    game_status_text = Column(String(40), nullable=True)

    home_team_id = Column(Integer, ForeignKey("nba_teams.id"), nullable=True)
    away_team_id = Column(Integer, ForeignKey("nba_teams.id"), nullable=True)

    home_score = Column(Integer, nullable=True)
    away_score = Column(Integer, nullable=True)
    home_wins = Column(Integer, nullable=True)
    home_losses = Column(Integer, nullable=True)
    away_wins = Column(Integer, nullable=True)
    away_losses = Column(Integer, nullable=True)

    period = Column(Integer, nullable=True)            # final period reached (4, 5=OT1, ...)
    duration = Column(String(10), nullable=True)       # e.g. "2:15", as NBA returns it

    arena_name = Column(String(80), nullable=True)
    arena_city = Column(String(60), nullable=True)
    arena_state = Column(String(20), nullable=True)
    attendance = Column(Integer, nullable=True)

    game_label = Column(String(60), nullable=True)      # e.g. "Round 1" — playoffs only
    game_sub_label = Column(String(60), nullable=True)
    series_text = Column(String(60), nullable=True)

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])


# ── Shared per-player-per-game mixin ─────────────────────────────────────────

class _PlayerGameStatMixin:
    """
    Shared identity columns for every "one row per player per game" box
    score variant below. All ten box-score endpoints in the original
    script return this same identity shape (gameId / teamId / personId /
    position) alongside a different bundle of statistics — this mixin
    exists purely to avoid repeating those four columns nine times.
    """
    id = Column(Integer, primary_key=True, autoincrement=True)
    # No index=True here: every table using this mixin declares a
    # UniqueConstraint("game_id", ...) with game_id as its leftmost
    # column, and InnoDB can already satisfy a game_id-only lookup from
    # that composite index's leftmost prefix — a separate single-column
    # index would just be redundant write overhead.
    game_id = Column(String(12), ForeignKey("nba_games.game_id"), nullable=False)
    team_id = Column(Integer, ForeignKey("nba_teams.id"), nullable=True)
    person_id = Column(Integer, ForeignKey("nba_players.person_id"), nullable=True)
    position = Column(String(10), nullable=True)


# ── V1 dedicated table: Traditional box score ───────────────────────────────

class BoxScoreTraditional(Base, _PlayerGameStatMixin):
    """
    The one box-score variant with a dedicated page in this pass — the
    "traditional" stat line most people mean by "box score." Source:
    BS_TRAD (boxscoretraditionalv3).
    """
    __tablename__ = "nba_box_score_traditional"
    __table_args__ = (UniqueConstraint("game_id", "person_id", name="uq_bs_trad_game_player"),)

    minutes = Column(String(10), nullable=True)   # "34:12" — kept as NBA's own string format
    field_goals_made = Column(Integer, nullable=True)
    field_goals_attempted = Column(Integer, nullable=True)
    field_goal_percentage = Column(Float, nullable=True)
    three_pointers_made = Column(Integer, nullable=True)
    three_pointers_attempted = Column(Integer, nullable=True)
    three_pointer_percentage = Column(Float, nullable=True)
    free_throws_made = Column(Integer, nullable=True)
    free_throws_attempted = Column(Integer, nullable=True)
    free_throw_percentage = Column(Float, nullable=True)
    rebounds_offensive = Column(Integer, nullable=True)
    rebounds_defensive = Column(Integer, nullable=True)
    rebounds_total = Column(Integer, nullable=True)
    assists = Column(Integer, nullable=True)
    steals = Column(Integer, nullable=True)
    blocks = Column(Integer, nullable=True)
    turnovers = Column(Integer, nullable=True)
    fouls_personal = Column(Integer, nullable=True)
    points = Column(Integer, nullable=True)
    plus_minus = Column(Float, nullable=True)


# ── Headless v1 tables: ingested now, dedicated UI is the fast-follow ───────

class BoxScoreAdvanced(Base, _PlayerGameStatMixin):
    __tablename__ = "nba_box_score_advanced"
    __table_args__ = (UniqueConstraint("game_id", "person_id", name="uq_bs_adv_game_player"),)

    minutes = Column(String(10), nullable=True)
    estimated_offensive_rating = Column(Float, nullable=True)
    offensive_rating = Column(Float, nullable=True)
    estimated_defensive_rating = Column(Float, nullable=True)
    defensive_rating = Column(Float, nullable=True)
    estimated_net_rating = Column(Float, nullable=True)
    net_rating = Column(Float, nullable=True)
    assist_percentage = Column(Float, nullable=True)
    assist_to_turnover = Column(Float, nullable=True)
    assist_ratio = Column(Float, nullable=True)
    offensive_rebound_percentage = Column(Float, nullable=True)
    defensive_rebound_percentage = Column(Float, nullable=True)
    rebound_percentage = Column(Float, nullable=True)
    turnover_ratio = Column(Float, nullable=True)
    effective_field_goal_percentage = Column(Float, nullable=True)
    true_shooting_percentage = Column(Float, nullable=True)
    usage_percentage = Column(Float, nullable=True)
    estimated_usage_percentage = Column(Float, nullable=True)
    estimated_pace = Column(Float, nullable=True)
    pace = Column(Float, nullable=True)
    pace_per_40 = Column(Float, nullable=True)
    possessions = Column(Float, nullable=True)
    pie = Column(Float, nullable=True)


class BoxScoreMisc(Base, _PlayerGameStatMixin):
    __tablename__ = "nba_box_score_misc"
    __table_args__ = (UniqueConstraint("game_id", "person_id", name="uq_bs_misc_game_player"),)

    minutes = Column(String(10), nullable=True)
    points_off_turnovers = Column(Integer, nullable=True)
    points_second_chance = Column(Integer, nullable=True)
    points_fast_break = Column(Integer, nullable=True)
    points_paint = Column(Integer, nullable=True)
    opp_points_off_turnovers = Column(Integer, nullable=True)
    opp_points_second_chance = Column(Integer, nullable=True)
    opp_points_fast_break = Column(Integer, nullable=True)
    opp_points_paint = Column(Integer, nullable=True)
    blocks = Column(Integer, nullable=True)
    blocks_against = Column(Integer, nullable=True)
    fouls_personal = Column(Integer, nullable=True)
    fouls_drawn = Column(Integer, nullable=True)


class BoxScoreScoring(Base, _PlayerGameStatMixin):
    __tablename__ = "nba_box_score_scoring"
    __table_args__ = (UniqueConstraint("game_id", "person_id", name="uq_bs_score_game_player"),)

    minutes = Column(String(10), nullable=True)
    pct_field_goals_attempted_2pt = Column(Float, nullable=True)
    pct_field_goals_attempted_3pt = Column(Float, nullable=True)
    pct_points_2pt = Column(Float, nullable=True)
    pct_points_midrange_2pt = Column(Float, nullable=True)
    pct_points_3pt = Column(Float, nullable=True)
    pct_points_fast_break = Column(Float, nullable=True)
    pct_points_free_throw = Column(Float, nullable=True)
    pct_points_off_turnovers = Column(Float, nullable=True)
    pct_points_paint = Column(Float, nullable=True)
    pct_assisted_2pt = Column(Float, nullable=True)
    pct_unassisted_2pt = Column(Float, nullable=True)
    pct_assisted_3pt = Column(Float, nullable=True)
    pct_unassisted_3pt = Column(Float, nullable=True)
    pct_assisted_fgm = Column(Float, nullable=True)
    pct_unassisted_fgm = Column(Float, nullable=True)


class BoxScoreUsage(Base, _PlayerGameStatMixin):
    __tablename__ = "nba_box_score_usage"
    __table_args__ = (UniqueConstraint("game_id", "person_id", name="uq_bs_usage_game_player"),)

    minutes = Column(String(10), nullable=True)
    usage_percentage = Column(Float, nullable=True)
    pct_field_goals_made = Column(Float, nullable=True)
    pct_field_goals_attempted = Column(Float, nullable=True)
    pct_three_pointers_made = Column(Float, nullable=True)
    pct_three_pointers_attempted = Column(Float, nullable=True)
    pct_free_throws_made = Column(Float, nullable=True)
    pct_free_throws_attempted = Column(Float, nullable=True)
    pct_rebounds_offensive = Column(Float, nullable=True)
    pct_rebounds_defensive = Column(Float, nullable=True)
    pct_rebounds_total = Column(Float, nullable=True)
    pct_assists = Column(Float, nullable=True)
    pct_turnovers = Column(Float, nullable=True)
    pct_steals = Column(Float, nullable=True)
    pct_blocks = Column(Float, nullable=True)
    pct_blocks_allowed = Column(Float, nullable=True)
    pct_personal_fouls = Column(Float, nullable=True)
    pct_personal_fouls_drawn = Column(Float, nullable=True)
    pct_points = Column(Float, nullable=True)


class BoxScoreFourFactors(Base, _PlayerGameStatMixin):
    __tablename__ = "nba_box_score_fourfactors"
    __table_args__ = (UniqueConstraint("game_id", "person_id", name="uq_bs_four_game_player"),)

    minutes = Column(String(10), nullable=True)
    effective_field_goal_percentage = Column(Float, nullable=True)
    free_throw_attempt_rate = Column(Float, nullable=True)
    team_turnover_percentage = Column(Float, nullable=True)
    offensive_rebound_percentage = Column(Float, nullable=True)
    opp_effective_field_goal_percentage = Column(Float, nullable=True)
    opp_free_throw_attempt_rate = Column(Float, nullable=True)
    opp_team_turnover_percentage = Column(Float, nullable=True)
    opp_offensive_rebound_percentage = Column(Float, nullable=True)


class BoxScoreTracking(Base, _PlayerGameStatMixin):
    __tablename__ = "nba_box_score_tracking"
    __table_args__ = (UniqueConstraint("game_id", "person_id", name="uq_bs_track_game_player"),)

    minutes = Column(String(10), nullable=True)
    distance = Column(Float, nullable=True)
    rebound_chances_offensive = Column(Integer, nullable=True)
    rebound_chances_defensive = Column(Integer, nullable=True)
    rebound_chances_total = Column(Integer, nullable=True)
    touches = Column(Integer, nullable=True)
    secondary_assists = Column(Integer, nullable=True)
    free_throw_assists = Column(Integer, nullable=True)
    passes = Column(Integer, nullable=True)
    assists = Column(Integer, nullable=True)
    contested_field_goals_made = Column(Integer, nullable=True)
    contested_field_goals_attempted = Column(Integer, nullable=True)
    contested_field_goal_percentage = Column(Float, nullable=True)
    uncontested_field_goals_made = Column(Integer, nullable=True)
    uncontested_field_goals_attempted = Column(Integer, nullable=True)
    uncontested_field_goal_percentage = Column(Float, nullable=True)
    field_goal_percentage = Column(Float, nullable=True)
    defended_at_rim_field_goals_made = Column(Integer, nullable=True)
    defended_at_rim_field_goals_attempted = Column(Integer, nullable=True)
    defended_at_rim_field_goal_percentage = Column(Float, nullable=True)


class BoxScoreHustle(Base, _PlayerGameStatMixin):
    __tablename__ = "nba_box_score_hustle"
    __table_args__ = (UniqueConstraint("game_id", "person_id", name="uq_bs_hustle_game_player"),)

    minutes = Column(String(10), nullable=True)
    points = Column(Integer, nullable=True)
    contested_shots = Column(Integer, nullable=True)
    contested_shots_2pt = Column(Integer, nullable=True)
    contested_shots_3pt = Column(Integer, nullable=True)
    deflections = Column(Integer, nullable=True)
    charges_drawn = Column(Integer, nullable=True)
    screen_assists = Column(Integer, nullable=True)
    screen_assist_points = Column(Integer, nullable=True)
    loose_balls_recovered_offensive = Column(Integer, nullable=True)
    loose_balls_recovered_defensive = Column(Integer, nullable=True)
    loose_balls_recovered_total = Column(Integer, nullable=True)
    offensive_box_outs = Column(Integer, nullable=True)
    defensive_box_outs = Column(Integer, nullable=True)
    box_out_player_team_rebounds = Column(Integer, nullable=True)
    box_out_player_rebounds = Column(Integer, nullable=True)
    box_outs = Column(Integer, nullable=True)


class BoxScoreMatchup(Base):
    """
    Matchup data is per (defender, offensive player) pair, not one row per
    player — doesn't fit _PlayerGameStatMixin's one-row-per-player shape,
    so this table defines its own identity columns. Not every game has
    this data available (source: boxscorematchupsv3).
    """
    __tablename__ = "nba_box_score_matchup"
    __table_args__ = (
        UniqueConstraint("game_id", "defender_person_id", "offensive_person_id",
                          name="uq_bs_matchup_game_pair"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Same reasoning as _PlayerGameStatMixin above: uq_bs_matchup_game_pair
    # already leads with game_id, so no separate index is needed.
    game_id = Column(String(12), ForeignKey("nba_games.game_id"), nullable=False)
    team_id = Column(Integer, ForeignKey("nba_teams.id"), nullable=True)
    defender_person_id = Column(Integer, ForeignKey("nba_players.person_id"), nullable=False)
    defender_position = Column(String(10), nullable=True)
    offensive_person_id = Column(Integer, ForeignKey("nba_players.person_id"), nullable=False)

    matchup_minutes = Column(String(10), nullable=True)
    matchup_minutes_sort = Column(Float, nullable=True)
    partial_possessions = Column(Float, nullable=True)
    pct_defender_total_time = Column(Float, nullable=True)
    pct_offensive_total_time = Column(Float, nullable=True)
    pct_total_time_both_on = Column(Float, nullable=True)
    switches_on = Column(Integer, nullable=True)
    player_points = Column(Integer, nullable=True)
    team_points = Column(Integer, nullable=True)
    matchup_assists = Column(Integer, nullable=True)
    matchup_potential_assists = Column(Integer, nullable=True)
    matchup_turnovers = Column(Integer, nullable=True)
    matchup_blocks = Column(Integer, nullable=True)
    matchup_field_goals_made = Column(Integer, nullable=True)
    matchup_field_goals_attempted = Column(Integer, nullable=True)
    matchup_field_goals_percentage = Column(Float, nullable=True)
    matchup_three_pointers_made = Column(Integer, nullable=True)
    matchup_three_pointers_attempted = Column(Integer, nullable=True)
    matchup_three_pointers_percentage = Column(Float, nullable=True)
    help_blocks = Column(Integer, nullable=True)
    help_field_goals_made = Column(Integer, nullable=True)
    help_field_goals_attempted = Column(Integer, nullable=True)
    help_field_goals_percentage = Column(Float, nullable=True)
    matchup_free_throws_made = Column(Integer, nullable=True)
    matchup_free_throws_attempted = Column(Integer, nullable=True)
    shooting_fouls = Column(Integer, nullable=True)


class BoxScoreDefensive(Base, _PlayerGameStatMixin):
    __tablename__ = "nba_box_score_defensive"
    __table_args__ = (UniqueConstraint("game_id", "person_id", name="uq_bs_def_game_player"),)

    matchup_minutes = Column(String(10), nullable=True)
    partial_possessions = Column(Float, nullable=True)
    switches_on = Column(Integer, nullable=True)
    player_points = Column(Integer, nullable=True)
    defensive_rebounds = Column(Integer, nullable=True)
    matchup_assists = Column(Integer, nullable=True)
    matchup_turnovers = Column(Integer, nullable=True)
    steals = Column(Integer, nullable=True)
    blocks = Column(Integer, nullable=True)
    matchup_field_goals_made = Column(Integer, nullable=True)
    matchup_field_goals_attempted = Column(Integer, nullable=True)
    matchup_field_goal_percentage = Column(Float, nullable=True)
    matchup_three_pointers_made = Column(Integer, nullable=True)
    matchup_three_pointers_attempted = Column(Integer, nullable=True)
    matchup_three_pointer_percentage = Column(Float, nullable=True)


# ── Play-by-play ─────────────────────────────────────────────────────────────

class PlayByPlayEvent(Base):
    """
    One row per action/event. Source: PBP (playbyplayv3). One of the three
    v1 dedicated-UI tables (see routers/nba_games.py's play-by-play tab).
    """
    __tablename__ = "nba_play_by_play"
    __table_args__ = (UniqueConstraint("game_id", "action_number", name="uq_pbp_game_action"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Same reasoning again: uq_pbp_game_action leads with game_id.
    game_id = Column(String(12), ForeignKey("nba_games.game_id"), nullable=False)
    action_number = Column(Integer, nullable=False)
    action_id = Column(Integer, nullable=True)

    period = Column(Integer, nullable=True)
    clock = Column(String(20), nullable=True)          # raw ISO-8601 duration, e.g. "PT11M32.00S"
    team_id = Column(Integer, ForeignKey("nba_teams.id"), nullable=True)
    team_tricode = Column(String(3), nullable=True)
    person_id = Column(Integer, ForeignKey("nba_players.person_id"), nullable=True)
    player_name = Column(String(80), nullable=True)

    action_type = Column(String(40), nullable=True)     # e.g. "shot", "rebound", "turnover"
    sub_type = Column(String(40), nullable=True)
    description = Column(Text, nullable=True)

    score_home = Column(Integer, nullable=True)
    score_away = Column(Integer, nullable=True)
    points_total = Column(Integer, nullable=True)

    shot_distance = Column(Integer, nullable=True)
    shot_result = Column(String(10), nullable=True)     # "Made" / "Missed"
    shot_value = Column(Integer, nullable=True)          # 2 or 3
    is_field_goal = Column(Boolean, nullable=True)
    location = Column(String(2), nullable=True)          # "h" / "v"
    shot_x = Column(Integer, nullable=True)               # xLegacy — shot-chart coordinate
    shot_y = Column(Integer, nullable=True)               # yLegacy — shot-chart coordinate
