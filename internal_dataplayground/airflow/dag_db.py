# airflow/dag_db.py
"""
Synchronous database session factory for use in Airflow DAG tasks.
Airflow tasks are synchronous — we use the sync SQLAlchemy engine here.
The async engine lives only in FastAPI (database.py).
"""
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from gcp_secrets import get_key


def get_sync_engine():
    mdb_json = json.loads(get_key("MariaDB"))
    url = f"mysql+pymysql://data_playground:{mdb_json['password']}@db:3306/jobs"
    return create_engine(url, pool_pre_ping=True)


def get_sync_session() -> Session:
    engine = get_sync_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()