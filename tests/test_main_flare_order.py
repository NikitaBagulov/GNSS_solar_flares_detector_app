from pathlib import Path

import main
from pipeline.run_config import RunConfig
from pipeline.runner import PipelineConfig


def _config(tmp_path):
    return PipelineConfig(
        start_date=__import__("datetime").date(2025, 11, 11),
        end_date=__import__("datetime").date(2025, 11, 11),
        min_flare_class="X1.0",
        state_json_path=tmp_path / "state.json",
        data_download_path=tmp_path / "data",
        run_config=RunConfig(),
    )


def test_run_pipeline_once_processes_each_flare_through_all_steps_before_next(monkeypatch, tmp_path):
    calls = []

    class DiscoveryResult:
        flare_keys = ["flare-a", "flare-b"]

    monkeypatch.setattr(main, "run_discovery", lambda config: DiscoveryResult())
    monkeypatch.setattr(
        main,
        "run_download_for_flare",
        lambda config, flare_key: calls.append(("download", flare_key)),
    )
    monkeypatch.setattr(
        main,
        "run_preprocessing_for_flare",
        lambda config, flare_key: calls.append(("preprocessing", flare_key)),
    )
    monkeypatch.setattr(
        main,
        "run_index_calculation_for_flare",
        lambda config, flare_key: calls.append(("index", flare_key)),
    )
    monkeypatch.setattr(
        main,
        "run_plotting_for_flare",
        lambda config, flare_key: calls.append(("plotting", flare_key)),
    )

    main.run_pipeline_once(
        _config(tmp_path),
        steps=["discovery", "preprocessing", "index", "plotting"],
    )

    assert calls == [
        ("download", "flare-a"),
        ("preprocessing", "flare-a"),
        ("index", "flare-a"),
        ("plotting", "flare-a"),
        ("download", "flare-b"),
        ("preprocessing", "flare-b"),
        ("index", "flare-b"),
        ("plotting", "flare-b"),
    ]


def test_run_pipeline_once_without_discovery_uses_known_flare_keys(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(main, "list_known_flare_keys", lambda config: ["flare-a"])
    monkeypatch.setattr(
        main,
        "run_index_calculation_for_flare",
        lambda config, flare_key: calls.append(("index", flare_key)),
    )

    main.run_pipeline_once(_config(tmp_path), steps=["index"])

    assert calls == [("index", "flare-a")]
