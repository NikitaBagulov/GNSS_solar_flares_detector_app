import numpy as np

from map_filters import empty_like_time_slice, filter_roti_time_slice, maybe_filter_roti_points


def test_empty_like_time_slice_preserves_2d_width():
    points = np.array([[1.0, 2.0, 3.0]])

    empty = empty_like_time_slice(points)

    assert empty.shape == (0, 3)
    assert empty.dtype == points.dtype


def test_filter_roti_time_slice_drops_invalid_zero_values_and_sorts():
    points = np.array(
        [
            [50.0, 30.0, 3.0],
            [51.0, 31.0, 0.0],
            [52.0, np.nan, 2.0],
            [53.0, 33.0, 1.0],
        ]
    )

    filtered = filter_roti_time_slice(points)

    assert filtered.tolist() == [[53.0, 33.0, 1.0], [50.0, 30.0, 3.0]]


def test_filter_roti_time_slice_handles_structured_arrays():
    dtype = [("lat", "f8"), ("lon", "f8"), ("vals", "f8"), ("site", "i4")]
    points = np.array(
        [
            (50.0, 30.0, 2.0, 1),
            (51.0, 31.0, 0.0, 2),
            (52.0, 32.0, 1.0, 3),
        ],
        dtype=dtype,
    )

    filtered = filter_roti_time_slice(points)

    assert filtered["vals"].tolist() == [1.0, 2.0]
    assert filtered.dtype == points.dtype


def test_maybe_filter_roti_points_only_filters_roti_sources():
    points = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]])

    assert maybe_filter_roti_points(points, "dtec_2_10") is points
    assert maybe_filter_roti_points(points, "map_roti").tolist() == [[3.0, 4.0, 5.0]]
