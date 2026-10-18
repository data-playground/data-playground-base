#!/usr/bin/env python3
"""
Applies the S1a edits to the two existing files, by exact-string replacement.
Run from the repo root:   python apply_s1a.py [--repo PATH]
Every 'old' snippet must match exactly once or nothing is written for that
file (all-or-nothing per file). Revert with: git checkout <the two files>.
"""
import argparse, pathlib, sys

DAG = "airflow/dags/soccer/life_os_soccer_ingest.py"
MODELS = "domains/soccer/models.py"

DAG_EDITS = [
('''from dag_db import fetch_all, fetch_one, execute, execute_many
from agents.soccer_agents import (''',
'''from dag_db import fetch_all, fetch_one, execute, execute_many
from agents.payload_codec import compress_json, decompress_json
from agents.soccer_agents import ('''),

('''def _store_raw(endpoint: str, fifa_competition_id, fifa_match_id, payload):
    execute(
        "INSERT INTO soccer_raw_payloads (endpoint, fifa_competition_id, fifa_match_id, payload) "
        "VALUES (%s, %s, %s, %s)",
        (endpoint, fifa_competition_id, fifa_match_id, json.dumps(payload)),
    )
''',
'''def _store_raw(endpoint: str, fifa_competition_id, fifa_match_id, payload):
    execute(
        "INSERT INTO soccer_raw_payloads "
        "(endpoint, fifa_competition_id, fifa_match_id, payload, payload_gz) "
        "VALUES (%s, %s, %s, %s, %s)",
        (endpoint, fifa_competition_id, fifa_match_id, json.dumps(payload), compress_json(payload)),
    )


def _load_raw_payload(row: dict):
    """
    Returns the parsed payload for a soccer_raw_payloads row selected as
    `payload_gz, IF(payload_gz IS NULL, payload, NULL) AS payload`.
    Prefers the compressed copy; falls back to the legacy column for rows
    written before payload_gz existed.
    """
    if row.get("payload_gz") is not None:
        return decompress_json(row["payload_gz"])
    legacy = row["payload"]
    return json.loads(legacy) if isinstance(legacy, str) else legacy
'''),

('''            "SELECT payload FROM soccer_raw_payloads "
            "WHERE endpoint = 'match_details' AND fifa_match_id = %s "''',
'''            "SELECT payload_gz, IF(payload_gz IS NULL, payload, NULL) AS payload "
            "FROM soccer_raw_payloads "
            "WHERE endpoint = 'match_details' AND fifa_match_id = %s "'''),

('''            payload = details_row["payload"]
            raw_details = json.loads(payload) if isinstance(payload, str) else payload
''',
'''            raw_details = _load_raw_payload(details_row)
'''),

('''            "SELECT payload FROM soccer_raw_payloads "
            "WHERE endpoint = 'match_events' AND fifa_match_id = %s "''',
'''            "SELECT payload_gz, IF(payload_gz IS NULL, payload, NULL) AS payload "
            "FROM soccer_raw_payloads "
            "WHERE endpoint = 'match_events' AND fifa_match_id = %s "'''),

('''                events_payload = events_row["payload"]
                raw_events = json.loads(events_payload) if isinstance(events_payload, str) else events_payload
''',
'''                raw_events = _load_raw_payload(events_row)
'''),
]

MODELS_EDITS = [
('''Float, JSON, ForeignKey, Index, UniqueConstraint,
)
from sqlalchemy.orm import relationship''',
'''Float, JSON, ForeignKey, Index, UniqueConstraint,
)
from sqlalchemy.dialects.mysql import MEDIUMBLOB
from sqlalchemy.orm import relationship'''),

('''    adding a new FIFA endpoint to ingest should never require a schema
    migration, only a new string value here.
    """
    __tablename__ = "soccer_raw_payloads"''',
'''    adding a new FIFA endpoint to ingest should never require a schema
    migration, only a new string value here.

    MIGRATION IN PROGRESS (S1a): `payload_gz` is the compressed copy
    (zlib-compressed compact JSON, see airflow/agents/payload_codec.py) and
    is the intended long-term store. `payload` (uncompressed JSON) is
    TRANSITIONAL: it is still dual-written so reverting the code stays safe,
    and it will be dropped in a later stage. Rows written before S1a have
    payload_gz NULL (no backfill yet); readers fall back to `payload`.
    """
    __tablename__ = "soccer_raw_payloads"'''),

('''        Index("ix_soccer_raw_payloads_lookup", "endpoint", "fifa_competition_id", "fifa_match_id"),
    )''',
'''        Index("ix_soccer_raw_payloads_lookup", "endpoint", "fifa_competition_id", "fifa_match_id"),
        Index("ix_soccer_raw_payloads_match_lookup", "fifa_match_id", "endpoint", "fetched_at"),
    )'''),

('''    payload = Column(JSON, nullable=False)
''',
'''    payload = Column(JSON, nullable=False)
    payload_gz = Column(MEDIUMBLOB, nullable=True)
'''),
]


def apply(path: pathlib.Path, edits):
    text = path.read_text(encoding="utf-8")
    for i, (old, new) in enumerate(edits, 1):
        n = text.count(old)
        if n != 1:
            sys.exit(f"STOP: {path} edit #{i} matched {n} times (expected 1) — "
                     f"file differs from what S1a was written against; nothing written to it.")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path} ({len(edits)} edits)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    repo = pathlib.Path(ap.parse_args().repo)
    apply(repo / DAG, DAG_EDITS)
    apply(repo / MODELS, MODELS_EDITS)
