# airflow/dag_db.py
"""
Synchronous database session factory for use in Airflow DAG tasks.
Airflow tasks are synchronous — we use the sync SQLAlchemy engine here.
The async engine lives only in FastAPI (database.py).
"""
import json

# from gcp_secrets import get_key
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker


def get_sync_engine():
    mdb_json = json.loads(os.environ.get("MARIA_DB"))
    url = f"mysql+pymysql://data_playground:{mdb_json['password']}@db:3306/jobs"
    return create_engine(url, pool_pre_ping=True)


def get_sync_session() -> Session:
    engine = get_sync_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
    

# airflow/dag_db.py
"""
Lightweight DB connector for Airflow DAG tasks.
Uses raw pymysql — no SQLAlchemy, no ORM, no version conflicts.
The full SQLAlchemy ORM lives only in FastAPI (database.py + models.py).
"""
import pymysql
import pymysql.cursors


def get_connection() -> pymysql.connections.Connection:
    mdb_json = json.loads(os.environ.get("MARIA_DB"))
    return pymysql.connect(
        host="db",
        user="data_playground",
        password=mdb_json["password"],
        database="jobs",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=False,
    )


def fetch_one(sql: str, params: tuple = ()) -> dict | None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()
    finally:
        conn.close()


def fetch_all(sql: str, params: tuple = ()) -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def execute(sql: str, params: tuple = ()) -> int:
    """Execute a write statement. Returns lastrowid."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
        return cur.lastrowid
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute_many(statements: list[tuple[str, tuple]]) -> None:
    """Execute multiple write statements in one transaction."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for sql, params in statements:
                cur.execute(sql, params)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()