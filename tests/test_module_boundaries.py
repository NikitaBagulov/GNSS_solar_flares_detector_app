from pathlib import Path

from pipeline_testkit import RunConfig, make_fixture_modules, run_pipeline_once


def test_module_boundaries_contracts(tmp_path):
    modules = make_fixture_modules(tmp_path)
    config = RunConfig(existing_data_policy="skip")

    result = run_pipeline_once(config, modules, context={"request_id": "abc"})
    outputs = result["outputs"]

    assert set(outputs.keys()) == {"download", "preprocess", "index", "plot"}
    assert "raw_path" in outputs["download"]
    assert "prep_path" in outputs["preprocess"]
    assert "index_path" in outputs["index"]
    assert "plot_path" in outputs["plot"]

    for module_name, payload in outputs.items():
        for key, value in payload.items():
            if key.endswith("_path"):
                assert Path(value).exists(), f"{module_name}:{key} must point to an existing artifact"


def test_smoke_pipeline_once_without_plot(tmp_path):
    modules = make_fixture_modules(tmp_path)
    config = RunConfig(existing_data_policy="skip", skip_modules={"plot"})

    result = run_pipeline_once(config, modules)
    outputs = result["outputs"]

    assert set(outputs.keys()) == {"download", "preprocess", "index"}
    assert (tmp_path / "raw" / "download.txt").exists()
    assert (tmp_path / "preprocessed" / "prep.txt").exists()
    assert (tmp_path / "indices" / "index.txt").exists()
    assert not (tmp_path / "plots" / "plot.txt").exists()
