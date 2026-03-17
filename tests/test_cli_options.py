from cli import parse_args


def test_cli_defaults():
    args = parse_args(["--start_date", "2025-01-01"])
    assert args.mode == "once"
    assert args.poll_interval_seconds == 3600
    assert args.existing_data_policy == "skip"
    assert args.skip_modules == []
    assert args.overwrite_modules == []
    assert args.validate_modules == []


def test_cli_new_flags_parsing():
    args = parse_args(
        [
            "--start_date",
            "2025-01-01",
            "--end_date",
            "2025-01-02",
            "--mode",
            "service",
            "--poll-interval-seconds",
            "120",
            "--existing-data-policy",
            "validate",
            "--skip-modules",
            "plot",
            "--overwrite-modules",
            "download",
            "index",
            "--validate-modules",
            "preprocess",
        ]
    )
    assert args.mode == "service"
    assert args.poll_interval_seconds == 120
    assert args.existing_data_policy == "validate"
    assert args.skip_modules == ["plot"]
    assert args.overwrite_modules == ["download", "index"]
    assert args.validate_modules == ["preprocess"]
