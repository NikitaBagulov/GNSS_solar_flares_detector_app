from datetime import date
from pathlib import Path

import pandas as pd

from pipeline.run_config import RunConfig
from pipeline.runner import PipelineConfig, run_discovery


class FakeTracker:
    state_file = Path("state.json")
    all_flares_file = Path("all_flares.csv")

    def __init__(self):
        self.updated = False
        self.synced = False

    def _update_flares_from_api(self):
        self.updated = True

    def sync_state_with_files(self):
        self.synced = True

    def _load_all_flares(self):
        return pd.DataFrame(
            [
                {"date": "2011-02-15", "flare_key": "old-flare"},
                {"date": "2014-10-24", "flare_key": "target-flare"},
                {"date": "2025-11-11", "flare_key": "future-flare"},
            ]
        )


def test_run_discovery_returns_only_flare_keys_in_requested_date_range(monkeypatch, tmp_path):
    tracker = FakeTracker()
    monkeypatch.setattr("pipeline.runner._load_tracker", lambda config: tracker)
    config = PipelineConfig(
        start_date=date(2014, 10, 23),
        end_date=date(2014, 10, 25),
        min_flare_class="X1.0",
        state_json_path=tmp_path / "state.json",
        data_download_path=tmp_path / "data",
        run_config=RunConfig(),
    )

    result = run_discovery(config)

    assert tracker.updated is True
    assert tracker.synced is True
    assert result.flare_keys == ["target-flare"]
