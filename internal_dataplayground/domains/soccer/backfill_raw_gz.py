# domains/soccer/backfill_raw_gz.py
"""
One-off ops script for the S1b stage of the soccer raw-first migration.
Acts on DATA in soccer_raw_payloads, never on project source files.

Self-contained, same pattern as domains/nba/seed_data.py: opens its own
pymysql connection (MARIA_DB env var, host "db"), imports only stdlib,
pymysql, and agents.payload_codec. Run inside a container that has the
project mounted, e.g.:

    docker exec -it life_os_airflow_scheduler \
        python /opt/airflow/project/domains/soccer/backfill_raw_gz.py --dry-run --batch 50

Modes (exactly one; default is backfill):
  (default)          Fill payload_gz for rows that only have the legacy
                     `payload`. Verifies each compressed copy round-trips
                     before writing; a row that fails is logged and skipped
                     (nothing written for it). Resumable and idempotent
                     (UPDATE only touches rows where payload_gz IS NULL).
                     Exits non-zero if any row was skipped.
  --dry-run          Same loop as backfill, no UPDATEs.
  --verify           Read-only pass over ALL rows: payload_gz must be set,
                     and when payload is not NULL it must equal
                     decompress_json(payload_gz). Prints ok / missing_gz /
                     mismatched (ids). Exits non-zero on any problem.
  --restore-payload  Rollback aid: for rows with payload IS NULL and
                     payload_gz NOT NULL, set payload from payload_gz.
                     Never touches rows that already have a payload.

Outside --restore-payload this script never writes the `payload` column.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/opt/airflow/project/airflow")

import pymysql
import pymysql.cursors

from agents.payload_codec import compress_json, decompress_json


def _connect():
    mdb_json = json.loads(os.environ["MARIA_DB"])
    return pymysql.connect(
        host="db", user="data_playground", password=mdb_json["password"],
        database="jobs", cursorclass=pymysql.cursors.DictCursor, autocommit=False,
    )


def _parse_legacy(value):
    """`payload` comes back as str (JSON column is LONGTEXT in MariaDB); tolerate bytes/parsed values."""
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def run_backfill(conn, batch: int, dry_run: bool) -> int:
    last_id = 0
    rows_done = skipped = 0
    bytes_in = bytes_out = 0
    skipped_ids: list[int] = []
    label = "DRY-RUN " if dry_run else ""

    while True:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, payload FROM soccer_raw_payloads "
                "WHERE payload_gz IS NULL AND payload IS NOT NULL AND id > %s "
                "ORDER BY id LIMIT %s",
                (last_id, batch),
            )
            rows = cur.fetchall()
        if not rows:
            break

        for row in rows:
            last_id = row["id"]
            try:
                raw_text = row["payload"]
                size_in = len(raw_text.encode("utf-8")) if isinstance(raw_text, str) else len(raw_text or b"")
                obj = _parse_legacy(raw_text)
                gz = compress_json(obj)
                if decompress_json(gz) != obj:
                    raise ValueError("round-trip mismatch")
            except Exception as exc:
                skipped += 1
                skipped_ids.append(row["id"])
                print(f"  SKIP id={row['id']}: {exc}", flush=True)
                continue

            if not dry_run:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE soccer_raw_payloads SET payload_gz=%s "
                        "WHERE id=%s AND payload_gz IS NULL",
                        (gz, row["id"]),
                    )
            rows_done += 1
            bytes_in += size_in
            bytes_out += len(gz)

        if not dry_run:
            conn.commit()
        print(f"{label}rows={rows_done} skipped={skipped} bytes_in={bytes_in} bytes_out={bytes_out} (last id {last_id})",
              flush=True)
        time.sleep(0.05)

    print(f"{label}DONE rows={rows_done} skipped={skipped} bytes_in={bytes_in} bytes_out={bytes_out}")
    if skipped_ids:
        print(f"Skipped ids: {skipped_ids}")
        return 1
    return 0


def run_verify(conn, batch: int) -> int:
    last_id = 0
    ok = 0
    missing_gz: list[int] = []
    mismatched: list[int] = []

    while True:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, payload_gz, payload FROM soccer_raw_payloads "
                "WHERE id > %s ORDER BY id LIMIT %s",
                (last_id, batch),
            )
            rows = cur.fetchall()
        if not rows:
            break
        for row in rows:
            last_id = row["id"]
            if row["payload_gz"] is None:
                missing_gz.append(row["id"])
                continue
            try:
                decoded = decompress_json(row["payload_gz"])
                if row["payload"] is not None and decoded != _parse_legacy(row["payload"]):
                    mismatched.append(row["id"])
                    continue
            except Exception:
                mismatched.append(row["id"])
                continue
            ok += 1

    print(f"ok={ok} missing_gz={len(missing_gz)} mismatched={len(mismatched)}")
    if missing_gz:
        print(f"missing_gz ids: {missing_gz}")
    if mismatched:
        print(f"mismatched ids: {mismatched}")
    return 1 if (missing_gz or mismatched) else 0


def run_restore_payload(conn, batch: int) -> int:
    last_id = 0
    restored = 0
    failed: list[int] = []

    while True:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, payload_gz FROM soccer_raw_payloads "
                "WHERE payload IS NULL AND payload_gz IS NOT NULL AND id > %s "
                "ORDER BY id LIMIT %s",
                (last_id, batch),
            )
            rows = cur.fetchall()
        if not rows:
            break
        for row in rows:
            last_id = row["id"]
            try:
                text = json.dumps(decompress_json(row["payload_gz"]))
            except Exception as exc:
                failed.append(row["id"])
                print(f"  FAIL id={row['id']}: {exc}", flush=True)
                continue
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE soccer_raw_payloads SET payload=%s WHERE id=%s AND payload IS NULL",
                    (text, row["id"]),
                )
            restored += 1
        conn.commit()
        print(f"restored={restored} failed={len(failed)} (last id {last_id})", flush=True)
        time.sleep(0.05)

    print(f"DONE restored={restored} failed={len(failed)}")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="S1b: backfill/verify/restore soccer_raw_payloads.payload_gz")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Backfill loop without writing.")
    mode.add_argument("--verify", action="store_true", help="Read-only check of ALL rows.")
    mode.add_argument("--restore-payload", action="store_true",
                      help="Refill payload from payload_gz for rows with payload IS NULL (rollback aid).")
    parser.add_argument("--batch", type=int, default=100, help="Rows per batch (default 100).")
    args = parser.parse_args()

    conn = _connect()
    try:
        if args.verify:
            code = run_verify(conn, args.batch)
        elif args.restore_payload:
            code = run_restore_payload(conn, args.batch)
        else:
            code = run_backfill(conn, args.batch, dry_run=args.dry_run)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
