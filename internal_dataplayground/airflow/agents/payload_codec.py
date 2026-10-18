# airflow/agents/payload_codec.py
"""
Compact, lossless JSON <-> bytes codec for soccer_raw_payloads.payload_gz
(S1a of the soccer raw-first migration).

stdlib only (json + zlib): importable from DAGs and, if ever needed, from
FastAPI code without crossing the DAG/FastAPI boundary (CONTRIBUTING.md).

Format: compact JSON (no spaces, real UTF-8 rather than \\uXXXX escapes)
-> zlib level 6. Round-trips any JSON-serializable value.
"""
import json
import zlib

_LEVEL = 6


def compress_json(obj) -> bytes:
    """JSON-serializable object -> zlib-compressed UTF-8 JSON bytes."""
    text = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    return zlib.compress(text.encode("utf-8"), _LEVEL)


def decompress_json(blob):
    """Inverse of compress_json(). Accepts bytes/bytearray/memoryview; raises on None."""
    if blob is None:
        raise ValueError("decompress_json() got None — no compressed payload to read")
    return json.loads(zlib.decompress(bytes(blob)).decode("utf-8"))
