from datetime import datetime

import numpy as np
import pytest

from index_functions.day_night_index import (
    RE_meters,
    _day_geometry,
    _subsolar_point,
    compute_day_night_index,
    great_circle_distance_vec,
)
from index_functions.gsflai import compute_gsflai_index, julian_day as gsflai_julian_day
from index_functions.isfai import compute_isfai_index, julian_day as isfai_julian_day


def test_great_circle_distance_zero_and_quarter_circumference():
    distances = great_circle_distance_vec(
        np.array([0.0, 0.0]),
        np.array([0.0, 90.0]),
        lat0=0.0,
        lon0=0.0,
    )

    assert distances[0] == pytest.approx(0.0)
    assert distances[1] == pytest.approx(np.pi * RE_meters / 2)


def test_day_night_index_returns_zero_for_invalid_or_one_sided_data():
    assert compute_day_night_index([], datetime(2025, 1, 1), debug=False) == 0.0
    assert compute_day_night_index([[0.0, 0.0, 1.0]], datetime(2025, 1, 1), debug=False) == 0.0


def test_day_geometry_weight_decreases_from_subsolar_point_to_terminator():
    quarter_circ = np.pi * RE_meters / 2.0
    weights, divisors = _day_geometry(
        np.array([0.0, quarter_circ / 2.0, quarter_circ]),
        min_cos=1e-6,
    )

    assert weights == pytest.approx([1.0, 0.5, 0.0])
    assert divisors[0] == pytest.approx(1.0)
    assert divisors[1] == pytest.approx(np.sqrt(0.5))
    assert divisors[2] == pytest.approx(1e-6)


def test_day_night_index_produces_finite_value_for_day_and_night_points():
    value = compute_day_night_index(
        [
            [0.0, 0.0, 3.0],
            [0.0, 180.0, 1.0],
            [20.0, 10.0, 2.5],
            [-20.0, -170.0, 0.5],
        ],
        datetime(2025, 3, 20, 12, 0, 0),
        debug=False,
    )

    assert np.isfinite(value)


def test_julian_day_helpers_agree_on_known_epoch():
    assert gsflai_julian_day(2000, 1, 1, 12, 0, 0) == pytest.approx(2451545.0)
    assert isfai_julian_day(2000, 1, 1, 12, 0, 0) == pytest.approx(2451545.0)


def test_gsflai_index_handles_too_few_or_degenerate_points():
    when = datetime(2025, 3, 20, 12, 0, 0)

    assert compute_gsflai_index([[0.0, 0.0, 1.0]], when) == 0.0
    assert compute_gsflai_index([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], when) == 0.0


def test_isfai_index_filters_invalid_points_and_returns_finite_value():
    value = compute_isfai_index(
        [
            [0.0, 0.0, 1.0],
            [10.0, 5.0, 2.0],
            [np.nan, 5.0, 100.0],
        ],
        datetime(2025, 3, 20, 12, 0, 0),
    )

    assert np.isfinite(value)


def test_day_night_variants_are_explicit_and_distinct():
    when = datetime(2025, 3, 20, 12, 0, 0)
    sub_lat, sub_lon = _subsolar_point(when)
    wrap = lambda lon: (lon + 180.0) % 360.0 - 180.0
    points = [
        [sub_lat, sub_lon, 1.0],
        [sub_lat, wrap(sub_lon + 80.0), 8.0],
        [sub_lat, wrap(sub_lon + 180.0), 0.5],
    ]

    legacy = compute_day_night_index(points, when, variant="legacy")
    weighted = compute_day_night_index(points, when, variant="distance_weight")
    corrected = compute_day_night_index(points, when, variant="distance_weight_cos")
    aggregate_cos = compute_day_night_index(points, when, variant="distance_weight_cos_sum")

    assert legacy != pytest.approx(weighted)
    assert corrected != pytest.approx(weighted)
    assert aggregate_cos != pytest.approx(corrected)
    assert all(np.isfinite(value) for value in (legacy, weighted, corrected, aggregate_cos))


def test_cosine_variant_supports_terminator_exclusion_and_epsilon_sensitivity():
    when = datetime(2025, 3, 20, 12, 0, 0)
    sub_lat, sub_lon = _subsolar_point(when)
    wrap = lambda lon: (lon + 180.0) % 360.0 - 180.0
    points = [
        [sub_lat, sub_lon, 1.0],
        [sub_lat, wrap(sub_lon + 89.9), 20.0],
        [sub_lat, wrap(sub_lon + 180.0), 0.5],
    ]

    included = compute_day_night_index(
        points, when, variant="distance_weight_cos", eps_abs=1e-3,
    )
    excluded = compute_day_night_index(
        points, when, variant="distance_weight_cos", eps_abs=1e-3,
        exclude_terminator_deg=5.0,
    )

    assert np.isfinite(included)
    assert np.isfinite(excluded)
    assert included != pytest.approx(excluded)
