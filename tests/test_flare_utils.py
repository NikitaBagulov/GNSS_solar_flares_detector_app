from datetime import datetime, timezone

import pandas as pd

from flare_utils import build_flare_key, get_flare_window, normalize_to_utc


def test_build_flare_key_includes_sanitized_class():
    key = build_flare_key(
        datetime(2025, 11, 11, 9, 49),
        datetime(2025, 11, 11, 10, 4, 30),
        datetime(2025, 11, 11, 10, 22, 13),
        "X 5.2",
    )

    assert key == "20251111T094900_100430_102213_X52"


def test_normalize_to_utc_marks_naive_datetime_as_utc():
    value = normalize_to_utc(datetime(2025, 1, 1, 12, 0, 0))

    assert value.tzinfo is not None
    assert value.utcoffset().total_seconds() == 0


def test_normalize_to_utc_preserves_aware_instant():
    original = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    assert normalize_to_utc(original) is original


def test_get_flare_window_expands_pandas_timestamps_in_utc():
    start, end = get_flare_window(
        pd.Timestamp("2025-11-11 09:49:00"),
        pd.Timestamp("2025-11-11 10:22:13"),
        window_minutes=15,
    )

    assert start == datetime(2025, 11, 11, 9, 34, 0, tzinfo=timezone.utc)
    assert end == datetime(2025, 11, 11, 10, 37, 13, tzinfo=timezone.utc)
