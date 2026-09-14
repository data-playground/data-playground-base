"""
Exercises life_os_medium_ingest.py's pure logic (_rows_to_sources_and_labels,
_build_upsert_statements) by actually importing the real file — not a
reimplementation — with `airflow` and `dag_db` faked out via sys.modules,
since neither is installed in this sandbox. DAG registration and the real
dag_db.execute_many/fetch_all calls are NOT exercised — those need a real
Airflow + MariaDB to test for real.
"""
import importlib.util
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]  # .../internal_dataplayground
sys.path.insert(0, str(REPO_ROOT))

# ── Fake `airflow` and `dag_db` just enough to import the DAG module ───────
class _FakeDAG:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


fake_airflow = types.ModuleType("airflow")
fake_airflow.DAG = _FakeDAG
sys.modules["airflow"] = fake_airflow

fake_operators_pkg = types.ModuleType("airflow.operators")
sys.modules["airflow.operators"] = fake_operators_pkg
fake_operators_python = types.ModuleType("airflow.operators.python")
fake_operators_python.PythonOperator = lambda *args, **kwargs: None
sys.modules["airflow.operators.python"] = fake_operators_python

recorded_execute_many_calls = []
fake_dag_db = types.ModuleType("dag_db")
fake_dag_db.execute_many = lambda statements: recorded_execute_many_calls.append(statements)
fake_dag_db.fetch_all = lambda sql, params=(): []  # unused directly by the tests below
sys.modules["dag_db"] = fake_dag_db

# ── Import the real file ────────────────────────────────────────────────────
DAG_PATH = REPO_ROOT / "airflow" / "dags" / "medium" / "life_os_medium_ingest.py"
spec = importlib.util.spec_from_file_location("life_os_medium_ingest", DAG_PATH)
dag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dag_module)  # __name__ != "__main__" here -> _register_dag() runs too, using the fakes above

from domains.medium.rss_ingest import FeedSourceType, ParsedArticle  # noqa: E402


def test_rows_to_sources_and_labels():
    rows = [
        {"source_type": "publication", "identifier": "example-publication", "label": "Example Pub"},
        {"source_type": "profile", "identifier": "@jsmith", "label": None},
        {"source_type": "bogus_type", "identifier": "whatever", "label": "Should be skipped"},
    ]
    sources, labels = dag_module._rows_to_sources_and_labels(rows)

    assert len(sources) == 2, "the unknown source_type row should be skipped, not raise"
    assert sources[0].type is FeedSourceType.PUBLICATION
    assert sources[0].identifier == "example-publication"
    assert sources[1].type is FeedSourceType.PROFILE

    assert labels[("publication", "example-publication")] == "Example Pub"
    assert labels[("profile", "@jsmith")] is None
    assert ("bogus_type", "whatever") not in labels
    print("_rows_to_sources_and_labels: OK")


def _fake_article(guid, source_type, source_identifier, tags):
    return ParsedArticle(
        guid=guid,
        title=f"Title for {guid}",
        url=f"https://medium.com/{guid}",
        author="Some Author",
        published_at=datetime(2026, 9, 13, tzinfo=timezone.utc),
        tags=tags,
        content_html="<p>placeholder</p>",
        source_type=source_type,
        source_identifier=source_identifier,
        fetched_at=datetime(2026, 9, 13, 1, 0, tzinfo=timezone.utc),
        raw_item="<item>placeholder</item>",
    )


def test_build_upsert_statements():
    labels = {("publication", "example-publication"): "Example Pub"}
    articles = [
        _fake_article("guid-1", FeedSourceType.PUBLICATION, "example-publication", ["tag-a", "tag-b"]),
        _fake_article("guid-2", FeedSourceType.PROFILE, "@jsmith", []),  # no label entry -> should be None
    ]

    statements = dag_module._build_upsert_statements(articles, labels)
    assert len(statements) == 2

    sql, params = statements[0]
    assert sql.startswith("INSERT INTO medium_articles")
    assert "ON DUPLICATE KEY UPDATE" in sql
    assert "guid=VALUES(guid)" not in sql, "guid is the key column, must not appear in the UPDATE clause"
    assert len(params) == len(dag_module._ARTICLE_COLUMNS)

    col_index = dag_module._ARTICLE_COLUMNS.index
    assert params[col_index("guid")] == "guid-1"
    assert params[col_index("source_label")] == "Example Pub"
    assert params[col_index("tags")] == '["tag-a", "tag-b"]'  # JSON-serialized, since raw SQL bypasses the ORM's JSON type

    _, params2 = statements[1]
    assert params2[col_index("source_label")] is None  # no matching labels entry
    print("_build_upsert_statements: OK")


def test_empty_articles_short_circuits():
    assert dag_module._build_upsert_statements([], {}) == []
    print("_build_upsert_statements: empty-list short-circuit OK")


if __name__ == "__main__":
    test_rows_to_sources_and_labels()
    test_build_upsert_statements()
    test_empty_articles_short_circuits()
    print("All tests passed.")
