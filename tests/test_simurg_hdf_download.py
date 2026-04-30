import datetime

from DataManager import DataManager
from download_functions import simurg_hdf
from download_functions.simurg_hdf import _format_bytes, _timeout_from_kwargs, download_simurg_hdf


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


def test_download_simurg_hdf_uses_urlretrieve_to_temp_file(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)

    def fake_urlretrieve(url, filename, reporthook=None):
        assert "simurg.iszf.irk.ru" in url
        if reporthook:
            reporthook(0, 0, 6)
            reporthook(1, 6, 6)
        filename.write_bytes(b"abcdef")
        return filename, {}

    monkeypatch.setattr(simurg_hdf, "urlretrieve", fake_urlretrieve)

    result = download_simurg_hdf(
        datetime.date(2025, 11, 11),
        manager,
        progress_interval=999,
    )

    assert result.read_bytes() == b"abcdef"


def test_download_simurg_hdf_restarts_existing_temp_file(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    temp_path = manager.get_download_path(
        "simurg_hdf",
        datetime.date(2025, 11, 11),
        "simurg_hdf_20251111.h5.tmp",
    )
    temp_path.write_bytes(b"abc")

    def fake_urlretrieve(url, filename, reporthook=None):
        assert not filename.exists()
        filename.write_bytes(b"fresh")
        return filename, {}

    monkeypatch.setattr(simurg_hdf, "urlretrieve", fake_urlretrieve)

    result = download_simurg_hdf(
        datetime.date(2025, 11, 11),
        manager,
        temp_path=temp_path,
        progress_interval=999,
    )

    assert result.read_bytes() == b"fresh"
