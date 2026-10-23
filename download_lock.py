"""Serialize download transactions and recovery for a shared data directory."""
import fcntl
import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

_mutex = threading.RLock()
_local = threading.local()


@contextmanager
def download_directory_lock(directory):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Reentrant in the owning thread (including signal cleanup), exclusive
    # across threads/processes. Never unlink the lock file: its inode is shared.
    with _mutex:
        held = getattr(_local, 'held', {})
        _local.held = held
        key = str(root)
        if key in held:
            yield
            return
        with (root / '.download.lock').open('a+b') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            held[key] = handle
            try:
                yield
            finally:
                del held[key]
                fcntl.flock(handle, fcntl.LOCK_UN)


def serialized_download(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with download_directory_lock(self.base_download_dir):
            return method(self, *args, **kwargs)
    return wrapped
