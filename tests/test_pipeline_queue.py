from pathlib import Path

from pipeline.queue import SQLiteJobQueue, stable_flare_key


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
