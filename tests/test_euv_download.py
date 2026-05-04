import datetime

import pandas as pd

from DataManager import DataManager
from download_functions import euv
from download_functions.euv import download_soho_sem


class FakeResponse:
    def __init__(self, status_code=200, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload

    def json(self):
        return self._payload


def _make_manager(tmp_path, monkeypatch):
    monkeypatch.setattr("DataManager.atexit.register", lambda *args, **kw: None)
    monkeypatch.setattr("DataManager.signal.signal", lambda *args, **kw: None)
    return DataManager(base_download_dir=tmp_path)


def test_download_soho_sem_uses_lasp_when_available(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    line = "0 0 0 15 0 0 0 0 0 0 0 0 1.25 2.5"

    def fake_get(url, **kwargs):
        assert "lasp.colorado.edu" in url
        return FakeResponse(text=line)

    monkeypatch.setattr(euv.requests, "get", fake_get)

    result = download_soho_sem(datetime.date(2024, 2, 22), manager)

    df = pd.read_csv(result)
    assert list(df.columns) == ["time", "flux_26_34", "flux_01_50"]
    assert df.loc[0, "flux_26_34"] == 1.25
    assert df.loc[0, "flux_01_50"] == 2.5


def test_download_soho_sem_falls_back_to_cdaweb_hapi(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "lasp.colorado.edu" in url:
            return FakeResponse(status_code=504, text="gateway timeout")
        assert "cdaweb.gsfc.nasa.gov/hapi/data" in url
        assert kwargs["params"]["id"] == "SOHO_CELIAS-SEM_15S"
        assert kwargs["params"]["parameters"] == "first_order_flux,central_order_flux"
        return FakeResponse(
            text=(
                "2024-02-22T00:00:00.000Z,3.0E+002e,4.0E+002e\n"
                "2024-02-22T00:00:15.000Z,3.5E+002e,4.5E+002e\n"
            )
        )

    monkeypatch.setattr(euv.requests, "get", fake_get)

    result = download_soho_sem(datetime.date(2024, 2, 22), manager)

    assert len(calls) == 2
    df = pd.read_csv(result)
    assert df["flux_26_34"].tolist() == [3.0, 3.5]
    assert df["flux_01_50"].tolist() == [4.0, 4.5]
