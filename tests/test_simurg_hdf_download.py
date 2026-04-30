import datetime

from DataManager import DataManager
from download_functions import simurg_hdf
from download_functions.simurg_hdf import _format_bytes, _timeout_from_kwargs, download_simurg_hdf


class FakeResponse:
    def __init__(self, chunks, headers=None, status_code=200):
        self._chunks = chunks
        self.headers = headers or {}
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield from self._chunks


def _make_manager(tmp_path, monkeypatch):
    monkeypatch.setattr("DataManager.atexit.register", lambda *args, **kw: None)
    monkeypatch.setattr("DataManager.signal.signal", lambda *args, **kw: None)
    return DataManager(base_download_dir=tmp_path)


def test_format_bytes():
    assert _format_bytes(10) == "10 B"
    assert _format_bytes(2048) == "2.0 KB"
    assert _format_bytes(3 * 1024 * 1024) == "3.0 MB"


def test_timeout_from_kwargs_defaults_to_connect_read_tuple():
    assert _timeout_from_kwargs({}) is None
    assert _timeout_from_kwargs({"timeout": 120}) == 120


def test_download_simurg_hdf_streams_chunks_to_temp_file(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    chunks = [b"abc", b"def"]

    def fake_get(url, timeout, stream, headers=None):
        assert stream is True
        assert timeout is None
        assert headers == {}
        return FakeResponse(chunks, headers={"content-length": "6"})

    monkeypatch.setattr(simurg_hdf.requests, "get", fake_get)

    result = download_simurg_hdf(
        datetime.date(2025, 11, 11),
        manager,
        progress_interval=999,
        chunk_size=2,
    )

    assert result.read_bytes() == b"abcdef"


def test_download_simurg_hdf_resumes_existing_temp_file(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    temp_path = manager.get_download_path(
        "simurg_hdf",
        datetime.date(2025, 11, 11),
        "simurg_hdf_20251111.h5.tmp",
    )
    temp_path.write_bytes(b"abc")
    calls = []

    def fake_get(url, timeout, stream, headers=None):
        calls.append(headers)
        return FakeResponse(
            [b"def"],
            headers={"content-length": "3", "content-range": "bytes 3-5/6"},
            status_code=206,
        )

    monkeypatch.setattr(simurg_hdf.requests, "get", fake_get)

    result = download_simurg_hdf(
        datetime.date(2025, 11, 11),
        manager,
        temp_path=temp_path,
        progress_interval=999,
    )

    assert calls == [{"Range": "bytes=3-"}]
    assert result.read_bytes() == b"abcdef"
