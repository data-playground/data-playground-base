import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]  # .../internal_dataplayground
sys.path.insert(0, str(REPO_ROOT))


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

sys.modules["airflow.operators"] = types.ModuleType("airflow.operators")
fake_operators_python = types.ModuleType("airflow.operators.python")
fake_operators_python.PythonOperator = lambda *args, **kwargs: None
sys.modules["airflow.operators.python"] = fake_operators_python

recorded_execute_calls = []


class _FakeIntegrityError(Exception):
    pass


def _fake_execute(sql, params=()):
    if "raise_duplicate" in str(params):
        raise _FakeIntegrityError(f"Duplicate entry '{params}' for key 'uq_medium_feed_source'")
    recorded_execute_calls.append((sql, params))
    return 1  # lastrowid


fake_dag_db = types.ModuleType("dag_db")
fake_dag_db.execute = _fake_execute
sys.modules["dag_db"] = fake_dag_db

DAG_PATH = REPO_ROOT / "airflow" / "dags" / "medium" / "life_os_medium_detect_source.py"
spec = importlib.util.spec_from_file_location("life_os_medium_detect_source", DAG_PATH)
dag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dag_module)

from domains.medium.rss_ingest import DetectionResult, FeedSourceType  # noqa: E402


class _FakeDagRun:
    def __init__(self, conf):
        self.conf = conf


def test_missing_raw_url_raises():
    try:
        dag_module.task_detect_source(dag_run=_FakeDagRun({}))
        raise AssertionError("expected ValueError for missing raw_url")
    except ValueError as exc:
        assert "raw_url" in str(exc)
    print("missing raw_url -> ValueError: OK")


def test_success_inserts_expected_row(monkeypatch_module_attr):
    fake_result = DetectionResult(
        matched=True,
        source_type=FeedSourceType.PUBLICATION,
        identifier="example-publication",
        sample_titles=["A", "B", "C"],
        attempted=[("https://medium.com/feed/example-publication", "OK — 3 article(s) found")],
    )
    monkeypatch_module_attr(dag_module, "identify_source", lambda raw_url: fake_result)

    recorded_execute_calls.clear()
    dag_module.task_detect_source(dag_run=_FakeDagRun({"raw_url": "https://medium.com/example-publication", "label": "My label"}))

    assert len(recorded_execute_calls) == 1
    sql, params = recorded_execute_calls[0]
    assert sql.strip().startswith("INSERT INTO medium_feed_sources")
    assert params[0] == "publication"
    assert params[1] == "example-publication"
    assert params[2] == "My label"
    assert params[3] is not None  # created_at timestamp
    print("success path -> correct INSERT params: OK")


def test_no_match_raises_with_attempted_detail(monkeypatch_module_attr):
    fake_result = DetectionResult(
        matched=False,
        source_type=None,
        identifier=None,
        sample_titles=[],
        attempted=[
            ("https://medium.com/feed/some-slug", "reachable but returned 0 articles"),
            ("https://medium.com/feed/@some-slug", "failed: 404 Not Found"),
        ],
    )
    monkeypatch_module_attr(dag_module, "identify_source", lambda raw_url: fake_result)

    try:
        dag_module.task_detect_source(dag_run=_FakeDagRun({"raw_url": "https://medium.com/some-slug"}))
        raise AssertionError("expected RuntimeError for no-match")
    except RuntimeError as exc:
        msg = str(exc)
        assert "some-slug" in msg
        assert "0 articles" in msg
        assert "404 Not Found" in msg
    print("no-match -> RuntimeError with attempted detail: OK")


def test_duplicate_propagates_as_task_failure(monkeypatch_module_attr):
    fake_result = DetectionResult(
        matched=True,
        source_type=FeedSourceType.PROFILE,
        identifier="raise_duplicate",  # triggers _fake_execute's simulated IntegrityError
        sample_titles=["X"],
        attempted=[],
    )
    monkeypatch_module_attr(dag_module, "identify_source", lambda raw_url: fake_result)

    try:
        dag_module.task_detect_source(dag_run=_FakeDagRun({"raw_url": "https://medium.com/@already-tracked"}))
        raise AssertionError("expected the fake IntegrityError to propagate")
    except _FakeIntegrityError:
        pass
    print("duplicate insert -> DB error propagates (fails the task): OK")


class _ModuleAttrPatcher:
    def __init__(self):
        self._restore = []

    def __call__(self, obj, name, value):
        self._restore.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in self._restore:
            setattr(obj, name, old)


if __name__ == "__main__":
    test_missing_raw_url_raises()

    patcher = _ModuleAttrPatcher()
    try:
        test_success_inserts_expected_row(patcher)
        test_no_match_raises_with_attempted_detail(patcher)
        test_duplicate_propagates_as_task_failure(patcher)
    finally:
        patcher.undo()

    print("All tests passed.")
