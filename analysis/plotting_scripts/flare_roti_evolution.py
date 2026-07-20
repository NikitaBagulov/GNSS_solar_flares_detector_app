#!/usr/bin/env python3
"""
Evolution of ROTI maps: before, at, and after flare peak.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import numpy as np

from .config import (
    PRODUCTS, PRODUCT_LABELS, FLARE_CLASSES,
    TIME_WINDOW_MINUTES, PLOT_FIGSIZE_SINGLE, PLOT_DPI,
    DEFAULT_RESULTS_DIR, DEFAULT_FLARES_CSV, DEFAULT_OUTPUT_DIR,
    OUTPUT_SUBDIRS, LABEL_FONT_SIZE, TICK_FONT_SIZE, TITLE_FONT_SIZE,
)
from .utils import (
    load_events, load_flare_catalog, find_flare_row,
    get_flare_time_window, load_hdf5_map, plot_global_map,
    find_nearest_map_time, save_figure, logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

N_SNAPSHOTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ROTI evolution before/at/after flare peak")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--flares-csv", type=Path, default=DEFAULT_FLARES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flare-classes", nargs="+", default=list(FLARE_CLASSES), choices=FLARE_CLASSES)
    parser.add_argument("--window-minutes", type=float, default=TIME_WINDOW_MINUTES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def select_snapshots(timestamps: list, peak_time: pd.Timestamp, n: int = N_SNAPSHOTS) -> list:
    if len(timestamps) <= n:
        return timestamps
    nearest = min(timestamps, key=lambda t: abs((t - peak_time).total_seconds()))
    idx = timestamps.index(nearest)
    half = (n - 1) // 2
    start = max(0, idx - half)
    end = min(len(timestamps), start + n)
    if end - start < n:
        start = max(0, end - n)
    return timestamps[start:end]


def plot_evolution_for_event(
    event: dict, results_dir: Path, catalog: pd.DataFrame,
    output_dir: Path, window_minutes: float, no_plots: bool,
) -> bool:
    flare_row = find_flare_row(event, catalog)
    if flare_row is None:
        logger.warning(f"[{event.get('name')}] No matching flare in catalog")
        return False

    peak_time = flare_row.get("peak_time")
    if pd.isna(peak_time):
        logger.warning(f"[{event.get('name')}] No peak time")
        return False

    time_window = get_flare_time_window(peak_time, window_minutes)
    event_name = event.get("name", "unknown")

    timestamps, map_data = load_hdf5_map(event, results_dir, "roti", time_window)
    if not timestamps:
        logger.warning(f"[{event_name}] No ROTI maps in window")
        return False

    snapshots = select_snapshots(timestamps, peak_time, N_SNAPSHOTS)
    if not snapshots:
        return False

    if no_plots:
        logger.info(f"[{event_name}] {len(snapshots)} snapshots available")
        return True

    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    if n == 1:
        axes = [axes]

    for ax, t in zip(axes, snapshots):
        points = map_data.get(t)
        if points is not None and points.size > 0 and points.dtype.names is not None:
            plot_global_map(ax, points, "roti", t, vmin=0.0, vmax=0.5)
        delta = (t - peak_time).total_seconds() / 60.0
        ax.set_title(f"t = {delta:+.0f} min", fontsize=TITLE_FONT_SIZE)

    fig.tight_layout()
    filename = f"roti_evolution_{peak_time:%H-%M-%S_UTC}.png"
    save_figure(fig, event_name, OUTPUT_SUBDIRS["maps"], filename, output_dir)
    logger.info(f"[{event_name}] Saved ROTI evolution ({n} snapshots)")
    return True


def main() -> None:
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    results_dir = args.results_dir.resolve()
    flares_csv = args.flares_csv.resolve()
    output_dir = args.output_dir.resolve()

    if not results_dir.exists() or not flares_csv.exists():
        logger.error("Results dir or flares CSV not found")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(results_dir)
    catalog = load_flare_catalog(flares_csv)
    logger.info(f"Loaded {len(events)} events, {len(catalog)} catalog flares")

    events_to_process = events[:args.max_events] if args.max_events else events
    total = success = 0
    for event in events_to_process:
        if event.get("class") not in args.flare_classes:
            continue
        ok = plot_evolution_for_event(event, results_dir, catalog, output_dir,
                                      args.window_minutes, args.no_plots)
        total += 1
        success += ok
        if not ok:
            logger.warning(f"[{event.get('name')}] Failed")
    logger.info(f"Done. {success}/{total} successful")


if __name__ == "__main__":
    main()
