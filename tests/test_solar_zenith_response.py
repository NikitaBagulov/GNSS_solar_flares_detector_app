import pandas as pd
import pytest

from analysis.plotting_scripts.flare_solar_zenith_response import aggregate_by_zenith_angle


def _points():
    return pd.DataFrame({
        "event": ["event"] * 4,
        "time": pd.to_datetime([
            "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z",
            "2025-01-01T00:00:00Z", "2025-01-01T00:01:00Z",
        ]),
        "product": ["roti"] * 4,
        "solar_zenith_angle_deg": [31.0] * 4,
        "response": [1.0, 1.0, 1.0, 9.0],
    })


def test_zenith_statistics_use_independent_time_slices():
    row = aggregate_by_zenith_angle(_points(), 5.0, 0.0, 180.0).iloc[0]

    assert row["time_count"] == 2
    assert row["point_count"] == 4
    assert row["median"] == pytest.approx(5.0)


def test_zenith_baseline_uses_first_time_slice_per_bin():
    row = aggregate_by_zenith_angle(
        _points(), 5.0, 0.0, 180.0, baseline_slices=1,
    ).iloc[0]

    assert row["median"] == pytest.approx(4.0)
