import multiprocessing
import importlib.util
import sys
from pathlib import Path
from download_lock import download_directory_lock

spec = importlib.util.spec_from_file_location('standalone_queue', Path(__file__).resolve().parents[1] / 'pipeline/queue.py')
queue_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = queue_module
spec.loader.exec_module(queue_module)
SQLiteJobQueue = queue_module.SQLiteJobQueue


def _wait_for_lock(root, entered):
    with download_directory_lock(root):
        entered.set()


def test_recovery_lock_waits_for_another_process(tmp_path):
    context = multiprocessing.get_context('spawn')
    entered = context.Event()
    with download_directory_lock(tmp_path):
        process = context.Process(target=_wait_for_lock, args=(tmp_path, entered))
        process.start()
        try:
            assert not entered.wait(.4)
        finally:
            pass
    try:
        assert entered.wait(10)
    finally:
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0


def test_discovery_preserves_backoff_and_never_revives_dead(tmp_path):
    queue = SQLiteJobQueue(tmp_path / 'q.sqlite3')
    job_id = queue.enqueue('download', target_date='2024-03-23', max_attempts=2)
    queue.claim()
    queue.fail(job_id, RuntimeError('network'), retry_delay_seconds=3600)
    queue.enqueue('download', target_date='2024-03-23')
    assert queue.claim() is None
    queue._connection().execute("UPDATE jobs SET available_at='2000-01-01' WHERE id=?", (job_id,))
    assert queue.claim().attempts == 2
    queue.fail(job_id, RuntimeError('network'), retry_delay_seconds=0)
    queue.enqueue('download', target_date='2024-03-23')
    assert queue.claim() is None
    assert queue._get(job_id).status == 'dead'
