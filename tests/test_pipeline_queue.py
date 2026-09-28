from datetime import date
from pathlib import Path

from pipeline.dispatcher import QueueDispatcher
from pipeline.queue import Job, SQLiteJobQueue, stable_flare_key
from pipeline.runner import should_plot_flare


def test_stable_flare_key_rejects_empty_values():
    assert stable_flare_key("  X/2025-01-01_X1.0  ") == "X/2025-01-01_X1.0"


def test_queue_deduplicates_and_claims_once(tmp_path: Path):
    queue = SQLiteJobQueue(tmp_path / "queue.sqlite3", worker_id="test")
    first = queue.enqueue("index", flare_key="X/2025-01-01_X1.0")
    second = queue.enqueue("index", flare_key="X/2025-01-01_X1.0")

    assert first == second
    job = queue.claim()
    assert job is not None
    assert job.flare_key == "X/2025-01-01_X1.0"
    assert job.attempts == 1
    assert queue.claim() is None


def test_queue_marks_dead_after_max_attempts(tmp_path: Path):
    queue = SQLiteJobQueue(tmp_path / "queue.sqlite3", worker_id="test")
    job_id = queue.enqueue("plot", flare_key="X/2025-01-01_X1.0", max_attempts=1)
    job = queue.claim()
    assert job is not None
    queue.fail(job_id, RuntimeError("broken"), retry_delay_seconds=0)
    assert queue.claim() is None
    row = queue._connection().execute("SELECT status, last_error FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row["status"] == "dead"
    assert "broken" in row["last_error"]


def test_plot_selection_includes_x_class_before_2019_and_all_classes_from_2019():
    assert should_plot_flare("flare", date(2018, 12, 31), "X1.0")
    assert not should_plot_flare("flare", date(2018, 12, 31), "M1.0")
    assert should_plot_flare("flare", date(2019, 1, 1), "C1.0")
    assert should_plot_flare("flare", date(2020, 1, 1), None)


def test_index_job_enqueues_plot_only_when_selection_rule_matches(monkeypatch, tmp_path):
    dispatcher = QueueDispatcher.__new__(QueueDispatcher)
    dispatcher.queue = SQLiteJobQueue(tmp_path / "queue.sqlite3", worker_id="test")
    dispatcher.max_attempts = 3
    dispatcher.config = object()
    job = Job(
        id=1,
        job_type="index",
        flare_key="flare",
        target_date="2018-12-31",
        payload={},
        status="running",
        attempts=1,
        max_attempts=3,
        last_error=None,
    )
    monkeypatch.setattr("pipeline.dispatcher.run_index_calculation_for_flare", lambda *_: type("Result", (), {"flare_keys": ["flare"]})())
    monkeypatch.setattr(dispatcher, "_should_plot", lambda _flare_key: False)

    dispatcher._execute(job)

    assert dispatcher.queue._connection().execute("SELECT COUNT(*) FROM jobs WHERE job_type='plot'").fetchone()[0] == 0
