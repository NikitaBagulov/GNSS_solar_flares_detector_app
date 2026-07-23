#!/usr/bin/env python3
"""
Analyze ionospheric response versus solar zenith angle.

The script scans event map files:
    <results_dir>/<event>/maps/map_<product>.h5

For every map point it calculates the solar zenith angle from UTC time,
latitude, and longitude, then aggregates the response in zenith-angle bins.

Supported products by default:
    roti, dtec_2_10, dtec_10_20, dtec_20_60

Outputs:
    response_vs_solar_zenith_points.csv   optional raw point table
    response_vs_solar_zenith_stats.csv    binned statistics
    response_vs_solar_zenith_<product>.png
    response_vs_solar_zenith_all.png

Example:
    python analyze_solar_zenith_response.py \
        --results-dir ./results \
        --output-dir ./outputs/solar_zenith \
        --products roti dtec_2_10 dtec_10_20 dtec_20_60 \
        --bin-width 5 \
        --response-mode absolute
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PRODUCTS = ("roti", "dtec_2_10", "dtec_10_20", "dtec_20_60")
PRODUCT_LABELS = {
    "roti": "ROTI (TECu/min)",
    "dtec_2_10": "dTEC 2\u201310 min (TECu)",
    "dtec_10_20": "dTEC 10\u201320 min (TECu)",
    "dtec_20_60": "dTEC 20\u201360 min (TECu)",
}

LOGGER = logging.getLogger("solar_zenith_response")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate ionospheric response versus solar zenith angle."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Root directory containing event subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/solar_zenith_response"),
        help="Directory for CSV tables and plots.",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=list(DEFAULT_PRODUCTS),
        help="Map products to process.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=5.0,
        help="Solar-zenith-angle bin width in degrees.",
    )
    parser.add_argument(
        "--response-mode",
        choices=("signed", "absolute", "squared"),
        default="absolute",
        help=(
            "Transformation of map values before aggregation: "
            "signed=value, absolute=abs(value), squared=value**2."
        ),
    )
    parser.add_argument(
        "--min-zenith-angle",
        type=float,
        default=0.0,
        help="Minimum solar zenith angle included, degrees.",
    )
    parser.add_argument(
        "--max-zenith-angle",
        type=float,
        default=180.0,
        help="Maximum solar zenith angle included, degrees.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=20,
        help="Minimum number of points required to plot a bin.",
    )
    parser.add_argument(
        "--save-points",
        action="store_true",
        help="Save the full point-level table; it may be large.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional limit on the number of event directories.",
    )
    parser.add_argument(
        "--events",
        nargs="+",
        default=None,
        help="Specific event names to process (e.g., 2011-08-09_X6.9). Overrides --max-events and --last-n-per-class.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        choices=("C", "M", "X"),
        help="Filter by flare class (C, M, X). Used together with --last-n-per-class.",
    )
    parser.add_argument(
        "--last-n-per-class",
        type=int,
        default=None,
        help="Take last N events per class (C, M, X). Overrides --max-events.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    return parser.parse_args()


def parse_hdf5_time(value: str) -> pd.Timestamp | None:
    """Parse an HDF5 dataset key as a UTC timestamp."""
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def julian_day(times: pd.DatetimeIndex) -> np.ndarray:
    """Convert UTC timestamps to Julian day."""
    unix_seconds = times.view("int64") / 1e9
    return unix_seconds / 86400.0 + 2440587.5


def solar_declination_and_subsolar_longitude(
    times: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate solar declination and subsolar longitude.

    The equations are adequate for zenith-angle binning and avoid requiring
    astropy or pvlib.
    """
    jd = julian_day(times)
    t = (jd - 2451545.0) / 36525.0

    mean_longitude = (
        280.46646 + 36000.76983 * t + 0.0003032 * t * t
    ) % 360.0
    mean_anomaly = (
        357.52911 + 35999.05029 * t - 0.0001537 * t * t
    ) % 360.0

    anomaly_rad = np.deg2rad(mean_anomaly)
    equation_center = (
        (1.914602 - 0.004817 * t - 0.000014 * t * t)
        * np.sin(anomaly_rad)
        + (0.019993 - 0.000101 * t) * np.sin(2.0 * anomaly_rad)
        + 0.000289 * np.sin(3.0 * anomaly_rad)
    )
    true_longitude = mean_longitude + equation_center

    obliquity = np.deg2rad(23.439291 - 0.0130042 * t)
    ecliptic_longitude = np.deg2rad(true_longitude)

    declination = np.arcsin(
        np.sin(obliquity) * np.sin(ecliptic_longitude)
    )

    right_ascension = np.arctan2(
        np.cos(obliquity) * np.sin(ecliptic_longitude),
        np.cos(ecliptic_longitude),
    )
    right_ascension_deg = np.rad2deg(right_ascension) % 360.0

    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    ) % 360.0

    greenwich_hour_angle = (gmst - right_ascension_deg) % 360.0
    subsolar_longitude = (-greenwich_hour_angle + 180.0) % 360.0 - 180.0

    return declination, np.deg2rad(subsolar_longitude)


def solar_zenith_angle_deg(
    times: pd.DatetimeIndex,
    latitudes_deg: np.ndarray,
    longitudes_deg: np.ndarray,
) -> np.ndarray:
    """Calculate solar zenith angle for arrays of UTC times and coordinates."""
    declination, subsolar_longitude = solar_declination_and_subsolar_longitude(times)

    latitude = np.deg2rad(latitudes_deg.astype(float))
    longitude = np.deg2rad(longitudes_deg.astype(float))
    hour_angle = longitude - subsolar_longitude

    sin_elevation = (
        np.sin(latitude) * np.sin(declination)
        + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    )
    sin_elevation = np.clip(sin_elevation, -1.0, 1.0)
    elevation_deg = np.rad2deg(np.arcsin(sin_elevation))
    return 90.0 - elevation_deg


def transform_response(values: np.ndarray, mode: str) -> np.ndarray:
    values = values.astype(float)
    if mode == "signed":
        return values
    if mode == "absolute":
        return np.abs(values)
    if mode == "squared":
        return values * values
    raise ValueError(f"Unknown response mode: {mode}")


def find_event_dirs(results_dir: Path) -> list[Path]:
    """Return directories that contain at least one map_*.h5 file."""
    event_dirs = {
        path.parent.parent
        for path in results_dir.rglob("maps/map_*.h5")
        if path.is_file()
    }
    return sorted(event_dirs)


def _parse_event_class(name: str) -> str | None:
    m = re.search(r"[_ ]([ABCMX])", name)
    return m.group(1) if m else None


def _filter_last_n_per_class(
    event_dirs: list[Path], n: int, classes: tuple[str, ...] | None = None
) -> list[Path]:
    by_class: dict[str, list[Path]] = {}
    for d in event_dirs:
        cls = _parse_event_class(d.name)
        if classes is not None and cls not in classes:
            continue
        if cls and cls in ("C", "M", "X"):
            by_class.setdefault(cls, []).append(d)
    target_classes = classes or ("C", "M", "X")
    result: list[Path] = []
    for cls in target_classes:
        selected = by_class.get(cls, [])
        result.extend(selected[-n:])
    return sorted(result, key=lambda p: p.name)


def read_product_points(
    event_dir: Path,
    product: str,
    response_mode: str,
) -> list[pd.DataFrame]:
    path = event_dir / "maps" / f"map_{product}.h5"
    if not path.exists():
        return []

    frames: list[pd.DataFrame] = []

    try:
        with h5py.File(path, "r") as file:
            if "data" not in file:
                LOGGER.warning("No 'data' group in %s", path)
                return []

            for time_key, dataset in file["data"].items():
                timestamp = parse_hdf5_time(time_key)
                if timestamp is None:
                    LOGGER.debug("Skipping invalid time key %s in %s", time_key, path)
                    continue

                try:
                    points = dataset[:]
                except Exception as exc:
                    LOGGER.warning("Could not read %s/%s: %s", path, time_key, exc)
                    continue

                if points.size == 0 or points.dtype.names is None:
                    continue

                required = {"lat", "lon", "vals"}
                if not required.issubset(points.dtype.names):
                    LOGGER.warning(
                        "Dataset %s/%s lacks fields %s",
                        path,
                        time_key,
                        sorted(required),
                    )
                    continue

                lat = np.asarray(points["lat"], dtype=float)
                lon = np.asarray(points["lon"], dtype=float)
                raw = np.asarray(points["vals"], dtype=float)

                valid = (
                    np.isfinite(lat)
                    & np.isfinite(lon)
                    & np.isfinite(raw)
                    & (lat >= -90.0)
                    & (lat <= 90.0)
                    & (lon >= -180.0)
                    & (lon <= 180.0)
                )
                if not np.any(valid):
                    continue

                lat = lat[valid]
                lon = lon[valid]
                raw = raw[valid]
                times = pd.DatetimeIndex([timestamp] * len(lat))

                zenith_angle = solar_zenith_angle_deg(times, lat, lon)
                response = transform_response(raw, response_mode)

                frames.append(
                    pd.DataFrame(
                        {
                            "event": event_dir.name,
                            "time": timestamp,
                            "product": product,
                            "latitude_deg": lat,
                            "longitude_deg": lon,
                            "solar_zenith_angle_deg": zenith_angle,
                            "raw_value": raw,
                            "response": response,
                        }
                    )
                )
    except OSError as exc:
        LOGGER.warning("Could not open %s: %s", path, exc)

    return frames


def collect_points(
    results_dir: Path,
    products: Iterable[str],
    response_mode: str,
    max_events: int | None,
    last_n_per_class: int | None,
    events: list[str] | None = None,
    classes: list[str] | None = None,
) -> pd.DataFrame:
    event_dirs = find_event_dirs(results_dir)

    if events is not None:
        wanted = set(events)
        event_dirs = [d for d in event_dirs if d.name in wanted]
    elif last_n_per_class is not None:
        event_dirs = _filter_last_n_per_class(
            event_dirs, last_n_per_class,
            classes=tuple(classes) if classes else None,
        )
    elif max_events is not None:
        event_dirs = event_dirs[:max_events]

    LOGGER.info("Found %d event directories", len(event_dirs))
    for d in event_dirs:
        LOGGER.info("  %s", d.name)

    all_frames: list[pd.DataFrame] = []

    for index, event_dir in enumerate(event_dirs, start=1):
        LOGGER.info("[%d/%d] %s", index, len(event_dirs), event_dir.name)
        for product in products:
            all_frames.extend(
                read_product_points(event_dir, product, response_mode)
            )

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


def aggregate_by_zenith_angle(
    points: pd.DataFrame,
    bin_width: float,
    min_zenith_angle: float,
    max_zenith_angle: float,
) -> pd.DataFrame:
    if bin_width <= 0:
        raise ValueError("--bin-width must be positive")
    if min_zenith_angle >= max_zenith_angle:
        raise ValueError("--min-zenith-angle must be below --max-zenith-angle")

    edges = np.arange(
        min_zenith_angle,
        max_zenith_angle + bin_width,
        bin_width,
        dtype=float,
    )
    if edges[-1] < max_zenith_angle:
        edges = np.append(edges, max_zenith_angle)

    selected = points[
        points["solar_zenith_angle_deg"].between(
            min_zenith_angle, max_zenith_angle, inclusive="both"
        )
    ].copy()

    selected["zenith_bin"] = pd.cut(
        selected["solar_zenith_angle_deg"],
        bins=edges,
        include_lowest=True,
        right=False,
    )

    grouped = (
        selected.groupby(["product", "zenith_bin"], observed=True)["response"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )

    grouped["zenith_left_deg"] = grouped["zenith_bin"].apply(
        lambda iv: float(iv.left)
    ).astype(float)
    grouped["zenith_right_deg"] = grouped["zenith_bin"].apply(
        lambda iv: float(iv.right)
    ).astype(float)
    grouped["zenith_center_deg"] = (
        grouped["zenith_left_deg"] + grouped["zenith_right_deg"]
    ) / 2.0
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))

    return grouped.drop(columns="zenith_bin").sort_values(
        ["product", "zenith_center_deg"]
    )


def ylabel_for_mode(product: str, response_mode: str) -> str:
    base = PRODUCT_LABELS.get(product, product)
    if response_mode == "signed":
        return f"Mean response: {base}"
    if response_mode == "absolute":
        return f"Mean absolute response: {base}"
    return f"Mean squared response: {base}\u00b2"


def plot_one_product(
    stats: pd.DataFrame,
    product: str,
    output_dir: Path,
    response_mode: str,
    min_count: int,
    event_name: str | None = None,
) -> None:
    data = stats[
        (stats["product"] == product) & (stats["count"] >= min_count)
    ].copy()
    if data.empty:
        LOGGER.warning("No sufficiently populated bins for %s", product)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = data["zenith_center_deg"].to_numpy()
    y = data["mean"].to_numpy()
    counts = data["count"].to_numpy()

    # Connector line
    ax.plot(x, y, color="grey", linewidth=1.2, alpha=0.6, zorder=2)
    # Scatter points colored by count using plasma colormap
    sc = ax.scatter(x, y, c=counts, cmap="plasma", s=60, edgecolors="grey",
                    linewidths=0.6, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, label="Number of observations")
    cbar.ax.tick_params(labelsize=11)

    ax.axvline(90.0, linestyle="--", linewidth=1.2, color="tab:red", alpha=0.7, label="Horizon (SZA = 90\u00b0)")
    ax.set_xlabel("Solar zenith angle (degrees)")
    ax.set_xlim(0.0, 180.0)
    ax.set_xticks(np.arange(0.0, 181.0, 30.0))
    ax.set_ylabel(ylabel_for_mode(product, response_mode))
    suffix = f" — {event_name}" if event_name else ""
    ax.set_title(f"{PRODUCT_LABELS.get(product, product)} vs solar zenith angle{suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    label = f"_{event_name}" if event_name else ""
    fig.savefig(
        output_dir / f"response_vs_solar_zenith_{product}{label}.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_all_products(
    stats: pd.DataFrame,
    products: Iterable[str],
    output_dir: Path,
    response_mode: str,
    min_count: int,
    event_name: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    plotted = False
    plasma = plt.cm.plasma
    product_list = [p for p in products]

    for i, product in enumerate(product_list):
        data = stats[
            (stats["product"] == product) & (stats["count"] >= min_count)
        ].copy()
        if data.empty:
            continue

        scale = data["mean"].abs().max()
        if not np.isfinite(scale) or scale == 0:
            continue

        color = plasma(i / max(len(product_list) - 1, 1))
        ax.plot(
            data["zenith_center_deg"],
            data["mean"] / scale,
            marker="o",
            linewidth=1.7,
            color=color,
            label=PRODUCT_LABELS.get(product, product),
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        LOGGER.warning("No data available for combined normalized plot")
        return

    ax.axvline(90.0, linestyle="--", linewidth=1.2, color="tab:red", alpha=0.7, label="Horizon (SZA = 90\u00b0)")
    ax.set_xlabel("Solar zenith angle (degrees)")
    ax.set_xlim(0.0, 180.0)
    ax.set_xticks(np.arange(0.0, 181.0, 30.0))
    ax.set_ylabel(f"Normalized mean {response_mode} response")
    suffix = f" — {event_name}" if event_name else ""
    ax.set_title(f"Normalized response versus solar zenith angle{suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    label = f"_{event_name}" if event_name else ""
    fig.savefig(
        output_dir / f"response_vs_solar_zenith_all{label}.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)


def select_event_dirs(
    results_dir: Path,
    events: list[str] | None,
    max_events: int | None,
    last_n_per_class: int | None,
    classes: list[str] | None,
) -> list[Path]:
    """Select event directories using the command-line filters."""
    event_dirs = find_event_dirs(results_dir)

    if events is not None:
        wanted = set(events)
        event_dirs = [d for d in event_dirs if d.name in wanted]
        missing = sorted(wanted - {d.name for d in event_dirs})
        for event_name in missing:
            LOGGER.warning("Event directory with map data was not found: %s", event_name)
    elif last_n_per_class is not None:
        event_dirs = _filter_last_n_per_class(
            event_dirs,
            last_n_per_class,
            classes=tuple(classes) if classes else None,
        )
    elif max_events is not None:
        event_dirs = event_dirs[:max_events]

    return event_dirs


def write_event_outputs(
    points: pd.DataFrame,
    products: Iterable[str],
    output_dir: Path,
    response_mode: str,
    bin_width: float,
    min_zenith_angle: float,
    max_zenith_angle: float,
    min_count: int,
    save_points: bool,
    event_name: str,
) -> None:
    """Write CSV tables and plots for one event."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = aggregate_by_zenith_angle(
        points=points,
        bin_width=bin_width,
        min_zenith_angle=min_zenith_angle,
        max_zenith_angle=max_zenith_angle,
    )

    stats_path = output_dir / "response_vs_solar_zenith_stats.csv"
    stats.to_csv(stats_path, index=False)
    LOGGER.info("Saved %s", stats_path)

    if save_points:
        points_path = output_dir / "response_vs_solar_zenith_points.csv"
        points.to_csv(points_path, index=False)
        LOGGER.info("Saved %s", points_path)

    for product in products:
        plot_one_product(
            stats=stats,
            product=product,
            output_dir=output_dir,
            response_mode=response_mode,
            min_count=min_count,
            event_name=event_name,
        )

    plot_all_products(
        stats=stats,
        products=products,
        output_dir=output_dir,
        response_mode=response_mode,
        min_count=min_count,
        event_name=event_name,
    )

    LOGGER.info(
        "[%s] Done: %d points, %d statistical rows, output=%s",
        event_name,
        len(points),
        len(stats),
        output_dir,
    )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    event_dirs = select_event_dirs(
        results_dir=results_dir,
        events=args.events,
        max_events=args.max_events,
        last_n_per_class=args.last_n_per_class,
        classes=args.classes,
    )
    if not event_dirs:
        raise SystemExit(
            "No matching event directories with map files were found. "
            "Check --events and the results directory layout."
        )

    LOGGER.info("Processing %d events separately", len(event_dirs))
    successful = 0

    for index, event_dir in enumerate(event_dirs, start=1):
        event_name = event_dir.name
        LOGGER.info("[%d/%d] %s", index, len(event_dirs), event_name)

        points = collect_points(
            results_dir=results_dir,
            products=args.products,
            response_mode=args.response_mode,
            max_events=None,
            last_n_per_class=None,
            events=[event_name],
            classes=None,
        )
        if points.empty:
            LOGGER.warning(
                "[%s] No valid map points; skipping this event",
                event_name,
            )
            continue

        write_event_outputs(
            points=points,
            products=args.products,
            output_dir=output_dir / event_name,
            response_mode=args.response_mode,
            bin_width=args.bin_width,
            min_zenith_angle=args.min_zenith_angle,
            max_zenith_angle=args.max_zenith_angle,
            min_count=args.min_count,
            save_points=args.save_points,
            event_name=event_name,
        )
        successful += 1

    if successful == 0:
        raise SystemExit(
            "No valid map points were found for any selected event. Check the "
            "HDF5 group 'data' and structured fields lat/lon/vals."
        )

    LOGGER.info(
        "Done: %d/%d events produced outputs under %s",
        successful,
        len(event_dirs),
        output_dir,
    )


if __name__ == "__main__":
    main()
