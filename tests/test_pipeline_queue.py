from datetime import date
from pathlib import Path
import threading

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


def test_stale_running_job_is_requeued_when_attempts_remain(tmp_path: Path):
    queue = SQLiteJobQueue(tmp_path / "queue.sqlite3", worker_id="test")
    job_id = queue.enqueue("download", target_date="2025-01-01", max_attempts=3)
    assert queue.claim() is not None
    queue._connection().execute(
        "UPDATE jobs SET locked_at='2000-01-01T00:00:00+00:00' WHERE id=?",
        (job_id,),
    )

    assert queue.recover_stale(timeout_seconds=60) == 1
    row = queue._connection().execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row["status"] == "pending"
    retried = queue.claim()
    assert retried is not None
    assert retried.id == job_id
    assert retried.attempts == 2


def test_dispatcher_recovers_and_checks_queue_before_discovery():
    calls = []
    stop_event = threading.Event()

    class EmptyQueue:
        def recover_stale(self, _timeout_seconds):
            calls.append("recover")
            return 0

        def claim(self):
            calls.append("claim")
            return None

    dispatcher = QueueDispatcher.__new__(QueueDispatcher)
    dispatcher.queue = EmptyQueue()

    def discover():
        calls.append("discover")
        stop_event.set()
        return 0

    dispatcher.discover = discover
    dispatcher.run(stop_event, poll_seconds=0, discovery_interval_seconds=1800)

    assert calls == ["recover", "claim", "discover"]


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
