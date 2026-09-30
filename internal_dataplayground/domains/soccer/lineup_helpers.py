# domains/soccer/lineup_helpers.py
"""
domains/soccer/lineup_helpers.py

Pure data-prep helpers for rendering a match's lineup as pitch-diagram
tokens. Kept out of routers/soccer.py deliberately — this is
presentation-shaping math (row/position layout), not routing logic, and
folding it into the router would make that file harder to read and push
it toward GOVERNANCE's 300-line router ceiling for no good reason.

FORMATION ROW ORDERING — position codes confirmed across several real
matches (Bournemouth/Brentford 4-2-3-1, Crystal Palace 3-4-2-1, Chelsea
4-2-3-1): 0=GK, 1=DEF, 2=AM/wide/wing-back (a broad catch-all — see
below), 3=FWD, 6=DM.

KNOWN LIMITATION, confirmed directly against Crystal Palace's real
3-4-2-1 lineup (2026-09-16): FIFA's Position field cannot distinguish
between different tactical bands that both happen to fall under the
generic code 2. Crystal Palace had exactly 6 players at Position=2 and
ZERO at Position=6 (DM) — meaning the formation's "4" band (2 wing-backs
+ 2 central mid) and "2" band (attacking mid) were BOTH coded as plain
"2", with nothing in the data saying which of those 6 players belongs to
which band. Compare to a 4-2-3-1 (Bournemouth/Brentford/Chelsea), where
that same span cleanly splits into Position=6 (the "2" DM band) and
Position=2 (the "3" AM band) — two different codes, no ambiguity.

_split_am_rows() below recovers the ROW STRUCTURE (how many rows, how
many players per row) for the leftover Position=2 group from the
Tactics string itself, once DEF/DM/FWD counts are subtracted out. It
CANNOT recover which specific player belongs in which resulting row —
nothing in the data says that — so it falls back to FIFA's own Players
array order for that assignment, which is a best-effort guess, not a
confirmed convention. This fixes the visual complaint (six players
crammed into one line instead of two rows of four and two) even though
individual placement within the split rows may occasionsally be off.
"""
from collections import defaultdict


def _parse_tactics(tactics: str | None) -> list[int] | None:
    """
    "3-4-2-1" -> [3, 4, 2, 1]. Returns None if missing or malformed —
    anything that doesn't cleanly parse to ints summing to 10 outfield
    players (the Tactics string never counts the goalkeeper).
    """
    if not tactics:
        return None
    try:
        parts = [int(p) for p in tactics.split("-")]
    except ValueError:
        return None
    if not parts or sum(parts) != 10:
        return None
    return parts


def _split_am_rows(
    am_players: list, tactics_parts: list[int] | None,
    def_count: int, fwd_count: int, dm_count: int,
) -> list[list]:
    """
    Recovers how many rows the leftover Position=2 group should split
    into, and how many players per row, from the Tactics string — see
    this module's docstring for the Crystal Palace example that
    motivated this. Falls back to one undivided row (the original,
    always-safe behavior) if the numbers don't cleanly reconcile —
    missing/malformed Tactics, or the remaining segment counts don't sum
    to the actual Position=2 headcount.
    """
    if not am_players:
        return []
    if not tactics_parts:
        return [am_players]

    remaining = list(tactics_parts)
    # Peel off the front (DEF) and back (FWD) segments if they match —
    # what's left is the span that Position=2 (and possibly Position=6,
    # handled next) needs to cover.
    if remaining and remaining[0] == def_count:
        remaining = remaining[1:]
    if remaining and remaining[-1] == fwd_count:
        remaining = remaining[:-1]
    # If DM players exist, one of the remaining segments IS the DM band
    # — remove the first match, since DM sits closest to defense. If DM
    # players don't exist (the Crystal Palace case), nothing to remove
    # here; the DM count's contribution to the formation number is
    # already folded into Position=2's larger headcount.
    if dm_count > 0:
        for i, seg in enumerate(remaining):
            if seg == dm_count:
                remaining = remaining[:i] + remaining[i + 1:]
                break

    if not remaining or sum(remaining) != len(am_players):
        return [am_players]

    rows = []
    cursor = 0
    for seg in remaining:
        rows.append(am_players[cursor:cursor + seg])
        cursor += seg
    return rows


def build_pitch_tokens(lineup_rows: list, goals: list, bookings: list, tactics: str | None = None) -> list[dict]:
    """
    Takes one team's SoccerMatchLineup rows (starters and bench both may
    be passed in — only is_starter=True rows are placed on the pitch),
    that match's SoccerGoal/SoccerBooking rows, and that team's Tactics
    string (e.g. "3-4-2-1" — pass match.home_formation / match.away_formation),
    and returns a flat list of dicts ready for direct Jinja iteration:

        {
            "player": <SoccerMatchLineup row>,
            "top_pct": float, "left_pct": float,
            "is_gk": bool,
            "goal_count": int, "assist_count": int,
            "has_card": bool,
            "subbed_off": bool,
        }

    Rows are spaced evenly across however many position groups actually
    result (see _split_am_rows() for how a single Position=2 group can
    become two rows) — a formation with no DM, or no AM split needed,
    doesn't leave an empty gap in the middle of the pitch.
    """
    goal_counts: dict[str, int] = defaultdict(int)
    assist_counts: dict[str, int] = defaultdict(int)
    for g in goals:
        if g.fifa_player_id:
            goal_counts[g.fifa_player_id] += 1
        if g.fifa_assist_player_id:
            assist_counts[g.fifa_assist_player_id] += 1

    carded_ids = {b.fifa_player_id for b in bookings if b.fifa_player_id}

    starters = [p for p in lineup_rows if p.is_starter]

    gk_players    = [p for p in starters if p.position_code == 0]
    def_players   = [p for p in starters if p.position_code == 1]
    dm_players    = [p for p in starters if p.position_code == 6]
    am_players    = [p for p in starters if p.position_code == 2]
    fwd_players   = [p for p in starters if p.position_code == 3]
    other_players = [p for p in starters if p.position_code not in (0, 1, 2, 3, 6)]

    tactics_parts = _parse_tactics(tactics)
    am_rows = _split_am_rows(am_players, tactics_parts, len(def_players), len(fwd_players), len(dm_players))

    # Visual rows, own-goal to attacking-goal order (bottom of pitch to
    # top). Unconfirmed-code players (other_players) sit just ahead of
    # the AM band rather than jammed up front with strikers — a
    # reasonable "somewhere advanced but not the front line" default.
    row_groups = [g for g in ([gk_players, def_players, dm_players] + am_rows + [other_players, fwd_players]) if g]

    n_rows = len(row_groups)
    tokens = []
    for row_index, players_in_row in enumerate(row_groups):
        top_pct = 92 - row_index * (92 - 14) / (n_rows - 1) if n_rows > 1 else 50.0
        count = len(players_in_row)
        for i, p in enumerate(players_in_row):
            left_pct = (i + 1) * (100 / (count + 1))
            tokens.append({
                "player": p,
                "top_pct": round(top_pct, 1),
                "left_pct": round(left_pct, 1),
                "is_gk": p.position_code == 0,
                "goal_count": goal_counts.get(p.fifa_player_id, 0),
                "assist_count": assist_counts.get(p.fifa_player_id, 0),
                "has_card": p.fifa_player_id in carded_ids,
                "subbed_off": p.is_starter and not p.on_field_at_finish,
            })
    return tokens


def build_crest_url(fifa_team_id: str | None) -> str | None:
    """
    https://api.fifa.com/api/v3/picture/teams-{format}-{size}/{IdTeam}.
    The "sq-4" format/size guess is UNVERIFIED — no network access was
    available to confirm it resolves. The template's <img onerror=...>
    falls back to a colored-initials placeholder if this 404s or gets
    CORS-blocked, so a wrong guess here degrades gracefully rather than
    breaking the page. If it turns out not to work, the real fix is
    proxying these images through our own backend rather than the
    browser hotlinking FIFA's CDN directly.
    """
    if not fifa_team_id:
        return None
    return f"https://api.fifa.com/api/v3/picture/teams-sq-4/{fifa_team_id}"
