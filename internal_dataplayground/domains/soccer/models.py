# domains/soccer/models.py
"""
Soccer domain ORM models (WO#34).

Two-layer storage, per the design decision documented in the WO#34 report:

  SoccerRawPayload  — every FIFA API response is stored here verbatim as
                       JSON, keyed by endpoint + competition + optional
                       entity (match) id. This is the "raw data" layer
                       requested in WO#34 Step 1 Q4 — it makes adding a
                       new FIFA endpoint (e.g. standings, once a working
                       endpoint is confirmed — none was in the source
                       script) a matter of calling store_raw() with a new
                       `endpoint` string in the DAG. No migration required
                       to add a new endpoint.

  SoccerCompetition, SoccerMatch — a lean normalized layer parsed out of
                       the raw calendar/matches payload, covering just
                       what the browsing UI in routers/soccer.py needs
                       (competition name, teams, score, kickoff time,
                       status). Match-level detail and play-by-play event
                       data (from FIFA's /live and /timelines endpoints)
                       are intentionally NOT normalized in this first
                       pass — those payloads are deeply nested and
                       specific to match presentation rather than
                       browsing/filtering, so they stay in
                       SoccerRawPayload only, rendered as formatted JSON
                       in the match detail page. Normalizing them is
                       flagged as a follow-up in the WO#34 report Notes,
                       not done here.

FIFA's own IDs (IdCompetition, IdSeason, IdStage, IdMatch) are stored as
strings, not integers — one competition ID in the source script
(8tddm56zbasf57jkkay4kbf11, for the men's Euros) is alphanumeric, so an
Integer column would break the moment that competition is ever watched.
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, Date, Boolean, JSON, ForeignKey, Index, UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.base_model import Base


class SoccerCompetition(Base):
    """
    A watched FIFA competition. Rows here are the "watch list" the daily
    ingest DAG iterates over — adding a new competition (e.g. the
    Olympics, ID 512) is an INSERT here, not a code change. See
    airflow/dags/soccer/life_os_soccer_ingest.py's _SEED_COMPETITIONS for
    what ships watched by default.
    """
    __tablename__ = "soccer_competitions"

    id = Column(Integer, primary_key=True)

    # FIFA's own competition ID. String, not Integer — see module
    # docstring (the men's Euros ID is alphanumeric).
    fifa_competition_id = Column(String(64), unique=True, nullable=False)

    name = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)

    # Only used the first time a competition is ingested (no matches exist
    # for it yet) — after that, the DAG uses a rolling recent-past/near-
    # future window instead. Lets a newly-added competition backfill its
    # season/tournament without every daily run for every OTHER
    # competition also re-pulling from this same far-back date.
    #
    # UPDATED (2026-09-10): this is now also editable after the fact from
    # /soccer/settings, to go further back than the original backfill
    # covered. See life_os_soccer_ingest.py::_needs_backfill() — the DAG
    # re-checks this value against the earliest match currently on file
    # for the competition on every run, not just "does it have zero
    # matches." So setting this to an earlier date on an already-ingested
    # competition genuinely triggers a wider pull on the next run (a
    # one-time larger fetch until the gap closes, then it settles back to
    # the cheap rolling window).
    backfill_from_date = Column(Date, nullable=True)

    created_at = Column(DateTime, server_default=func.now())

    matches = relationship("SoccerMatch", back_populates="competition")

    def __repr__(self):
        return f"<SoccerCompetition {self.fifa_competition_id} {self.name!r}>"


class SoccerMatch(Base):
    """
    One match, normalized from the FIFA calendar/matches endpoint
    response. Upserted daily by the ingest DAG — score and status_label
    get updated in place as a match goes scheduled -> live -> finished.
    """
    __tablename__ = "soccer_matches"
    __table_args__ = (
        UniqueConstraint(
            "fifa_competition_id", "fifa_season_id", "fifa_stage_id", "fifa_match_id",
            name="uq_soccer_match_identity",
        ),
        Index("ix_soccer_matches_kickoff", "kickoff_at"),
    )

    id = Column(Integer, primary_key=True)

    competition_id = Column(Integer, ForeignKey("soccer_competitions.id"), nullable=False)

    # FIFA's own composite identity for a match — all four are required to
    # build the /live and /timelines detail URLs (see soccer_agents.py).
    fifa_competition_id = Column(String(64), nullable=False)
    fifa_season_id      = Column(String(64), nullable=False)
    fifa_stage_id       = Column(String(64), nullable=False)
    fifa_match_id       = Column(String(64), nullable=False)

    home_team_name = Column(String(255), nullable=True)
    away_team_name = Column(String(255), nullable=True)
    home_team_score = Column(Integer, nullable=True)
    away_team_score = Column(Integer, nullable=True)

    kickoff_at = Column(DateTime, nullable=True)

    # Raw FIFA MatchStatus integer code, preserved as-is (see
    # soccer_agents.py's _MATCH_STATUS_MAP — flagged there as unverified
    # against a live response).
    fifa_match_status_code = Column(Integer, nullable=True)
    # Best-effort mapped label: "scheduled" | "live" | "finished" | "unknown"
    status_label = Column(String(20), nullable=False, default="unknown")

    venue_name = Column(String(255), nullable=True)

    # Set once /live and /timelines have been fetched for this match, so
    # the DAG doesn't re-fetch detail/events for a match every single day
    # once it's finished. NULL means "never fetched." Mirrors the
    # streaming_fetched_at pattern in media_agents.py /
    # life_os_refresh_streaming_availability.py.
    details_fetched_at = Column(DateTime, nullable=True)

    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    competition = relationship("SoccerCompetition", back_populates="matches")

    def __repr__(self):
        return f"<SoccerMatch {self.home_team_name} vs {self.away_team_name} ({self.fifa_match_id})>"


class SoccerRawPayload(Base):
    """
    Verbatim FIFA API response storage — the "raw data" layer. See module
    docstring. `endpoint` is a free-text label, not an Enum, deliberately:
    adding a new FIFA endpoint to ingest should never require a schema
    migration, only a new string value here.
    """
    __tablename__ = "soccer_raw_payloads"
    __table_args__ = (
        Index("ix_soccer_raw_payloads_lookup", "endpoint", "fifa_competition_id", "fifa_match_id"),
    )

    id = Column(Integer, primary_key=True)

    # "competitions" | "matches" | "match_details" | "match_events" | ...
    # (future endpoints, e.g. "standings", just add a new string here)
    endpoint = Column(String(50), nullable=False)

    fifa_competition_id = Column(String(64), nullable=True)
    fifa_match_id = Column(String(64), nullable=True)  # null for competitions/matches-list payloads

    payload = Column(JSON, nullable=False)

    fetched_at = Column(DateTime, server_default=func.now())

    def __repr__(self):
        return f"<SoccerRawPayload {self.endpoint} comp={self.fifa_competition_id} match={self.fifa_match_id}>"


class SoccerSettings(Base):
    """
    Singleton settings row for the soccer domain — same get-or-default
    shape as HabitSettings elsewhere in this app (see routers/dashboard.py's
    `grace_result = await db.execute(select(HabitSettings).limit(1))`
    pattern, mirrored by
    domains/soccer/routers/soccer_settings.py::_get_or_create_settings()).

    Read by the daily ingest DAG at the start of each run
    (airflow/dags/soccer/life_os_soccer_ingest.py::_get_window_days(), via
    dag_db.py raw SQL — the DAG never imports this model directly, per
    the DAG/FastAPI boundary rule) to size its rolling fixtures/results
    pull window. Falls back to the DAG's own module-level defaults if
    this table is empty (e.g. before the FastAPI app has ever created a
    row) or doesn't exist yet.
    """
    __tablename__ = "soccer_settings"

    id = Column(Integer, primary_key=True)

    # Keep these defaults in sync with life_os_soccer_ingest.py's
    # ROLLING_WINDOW_PAST_DAYS / ROLLING_WINDOW_FUTURE_DAYS fallback
    # constants — they're the same "day one, nobody's touched settings
    # yet" values.
    window_past_days = Column(Integer, nullable=False, default=3)
    window_future_days = Column(Integer, nullable=False, default=60)

    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
