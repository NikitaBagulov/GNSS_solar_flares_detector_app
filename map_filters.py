from pathlib import Path

import numpy as np


def empty_like_time_slice(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.dtype.names is not None:
        return np.empty((0,), dtype=points.dtype)
    if points.ndim <= 1:
        return np.empty((0,), dtype=points.dtype)
    return np.empty((0, points.shape[1]), dtype=points.dtype)


def filter_roti_time_slice(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    if points.size == 0:
        return empty_like_time_slice(points)

    if points.dtype.names is not None:
        required_fields = ("lat", "lon", "vals")
        if not all(field in points.dtype.names for field in required_fields):
            return points

        mask = (
            np.isfinite(points["lat"])
            & np.isfinite(points["lon"])
            & np.isfinite(points["vals"])
            & (points["vals"] != 0)
        )
        return points[mask]

    if points.ndim < 2 or points.shape[1] < 3:
        return points

    mask = (
        np.isfinite(points[:, 0])
        & np.isfinite(points[:, 1])
        & np.isfinite(points[:, 2])
        & (points[:, 2] != 0)
    )
    return points[mask]


def maybe_filter_roti_points(points: np.ndarray, source_name: str | Path) -> np.ndarray:
    source_path = Path(source_name)
    source_tokens = {
        source_path.name,
        source_path.stem,
        str(source_name),
    }
    if "roti" not in source_tokens and "map_roti" not in source_tokens and "map_roti.h5" not in source_tokens:
        return points

    filtered_points = filter_roti_time_slice(points)
    if filtered_points.size == 0:
        return empty_like_time_slice(points)
    return filtered_points
