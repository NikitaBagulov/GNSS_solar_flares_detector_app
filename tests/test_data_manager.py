from pathlib import Path

import pandas as pd

from DataManager import DataManager


def _make_manager(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr("DataManager.atexit.register", lambda *args, **kw: None)
    monkeypatch.setattr("DataManager.signal.signal", lambda *args, **kw: None)
    return DataManager(base_download_dir=tmp_path, **kwargs)


def test_get_download_path_creates_date_source_directory(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)

    path = manager.get_download_path("goes_xray", pd.Timestamp("2025-11-11").date(), "goes.csv")

    assert path == tmp_path / "2025-11-11" / "goes_xray" / "goes.csv"
    assert path.parent.exists()


def test_csv_file_validation_requires_nonempty_readable_csv(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("time,xrsb\n2025-01-01,1.0\n", encoding="utf-8")

    assert manager._is_file_valid(csv_path, "goes_xray") is True
    assert manager._is_file_valid(tmp_path / "missing.csv", "goes_xray") is False


def test_download_by_date_commits_temp_file_to_final_path(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch, existing_data_policy="validate")

    def fake_download(date, data_manager, temp_path, **kwargs):
        Path(temp_path).write_text("time,xrsb\n2025-11-11,1.0\n", encoding="utf-8")
        return Path(temp_path)

    manager.register_download_function("goes_xray", fake_download)

    result = manager.download_by_date(pd.Timestamp("2025-11-11").date(), sources=["goes_xray"])
    final_path = tmp_path / "2025-11-11" / "goes_xray" / "goes_xray_20251111.csv"

    assert result["goes_xray"]["status"] == "success"
    assert final_path.exists()
    assert not final_path.with_suffix(".csv.tmp").exists()


def test_download_by_date_skip_policy_keeps_existing_file(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch, existing_data_policy="skip")
    target_date = pd.Timestamp("2025-11-11").date()
    final_path = manager.get_download_path("goes_xray", target_date, "goes_xray_20251111.csv")
    final_path.write_text("time,xrsb\nold,1.0\n", encoding="utf-8")

    def fake_download(**kwargs):
        raise AssertionError("download function should not be called for skip policy")

    manager.register_download_function("goes_xray", fake_download)

    result = manager.download_by_date(target_date, sources=["goes_xray"])

    assert result["goes_xray"]["status"] == "skipped"
    assert final_path.read_text(encoding="utf-8") == "time,xrsb\nold,1.0\n"
