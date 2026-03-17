from pathlib import Path

from pipeline_testkit import apply_data_policy, RunConfig, effective_policy, should_run_module


def _producer(path: Path, value: str):
    path.write_text(value)
    return value


def test_policy_skip(tmp_path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("keep")
    result = apply_data_policy(file_path, "skip", validator=lambda p: True, producer=lambda p: _producer(p, "new"))
    assert result["status"] == "skipped"
    assert file_path.read_text() == "keep"


def test_policy_overwrite(tmp_path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("old")
    result = apply_data_policy(file_path, "overwrite", validator=lambda p: True, producer=lambda p: _producer(p, "new"))
    assert result["status"] == "recreated"
    assert file_path.read_text() == "new"


def test_policy_validate_repair(tmp_path):
    file_path = tmp_path / "a.txt"
    file_path.write_text("broken")
    result = apply_data_policy(file_path, "validate", validator=lambda p: p.read_text() == "ok", producer=lambda p: _producer(p, "ok"))
    assert result["status"] == "repaired"
    assert file_path.read_text() == "ok"


def test_effective_policy_and_skip():
    config = RunConfig(
        existing_data_policy="skip",
        skip_modules={"plot"},
        overwrite_modules={"download"},
        validate_modules={"index"},
    )
    assert should_run_module("plot", config) is False
    assert effective_policy("download", config) == "overwrite"
    assert effective_policy("index", config) == "validate"
    assert effective_policy("preprocess", config) == "skip"
