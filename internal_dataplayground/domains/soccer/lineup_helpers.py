# domains/soccer/lineup_helpers.py
"""
domains/soccer/lineup_helpers.py

Pure data-prep helpers for rendering a match's lineup as pitch-diagram
tokens. Kept out of routers/soccer.py deliberately — this is
presentation-shaping math (row/position layout), not routing logic, and
folding it into the router would make that file harder to read and push
it toward GOVERNANCE's 300-line router ceiling for no good reason.

FORMATION ROW ORDERING — confirmed only against one real match
(Bournemouth vs Brentford, both 4-2-3-1, 2026-09-12). Position codes
seen: 0=GK, 1=DEF, 2=AM/wide, 3=FWD, 6=DM. The explicit ordering below
places these in actual on-pitch order (GK -> DEF -> DM -> AM -> FWD)
rather than numeric order, which would wrongly sort DM=6 after FWD=3.
Any position code not in this map falls into a single fallback row
placed between DEF and AM — a reasonable "somewhere in midfield" guess,
not a confirmed mapping. Extend _ROW_ORDER as new codes are confirmed
from future matches with different formations (back three, wing-backs,
etc. may use codes not yet seen).
"""
from collections import defaultdict

_ROW_ORDER = {0: 0, 1: 1, 6: 2, 2: 3, 3: 4}
_FALLBACK_ROW = 2.5  # unconfirmed codes only — sits between DEF and DM/AM


def build_pitch_tokens(lineup_rows: list, goals: list, bookings: list) -> list[dict]:
    """
    Takes one team's SoccerMatchLineup rows (starters and bench both may
    be passed in — only is_starter=True rows are placed on the pitch)
    plus that match's SoccerGoal/SoccerBooking rows (either team's, will
    be filtered to this team's players by ID lookup), and returns a
    flat list of dicts ready for direct Jinja iteration:

        {
            "player": <SoccerMatchLineup row>,
            "top_pct": float, "left_pct": float,
            "is_gk": bool,
            "goal_count": int, "assist_count": int,
            "has_card": bool,
            "subbed_off": bool,
        }

    Rows are spaced evenly across whatever position groups are ACTUALLY
    present for this lineup, not fixed slots — a formation with no DM
    (no code-6 players) doesn't leave an empty gap in the middle of the
    pitch.
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

    rows: dict[float, list] = defaultdict(list)
    for p in starters:
        row_key = _ROW_ORDER.get(p.position_code, _FALLBACK_ROW)
        rows[row_key].append(p)

    ordered_row_keys = sorted(rows.keys())
    n_rows = len(ordered_row_keys)

    tokens = []
    for row_index, row_key in enumerate(ordered_row_keys):
        # GK row sits closest to this team's own goal line (92%); the
        # frontmost row sits at 14%. Evenly spread across however many
        # rows are actually present.
        top_pct = 92 - row_index * (92 - 14) / (n_rows - 1) if n_rows > 1 else 50.0
        players_in_row = rows[row_key]
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
