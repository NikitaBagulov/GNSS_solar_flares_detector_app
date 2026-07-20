#!/usr/bin/env python3
"""
Hovmöller diagram: time vs latitude, color = ROTI.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .config import (
    FLARE_CLASSES, PLOT_FIGSIZE_SINGLE, PLOT_DPI,
    DEFAULT_RESULTS_DIR, DEFAULT_FLARES_CSV, DEFAULT_OUTPUT_DIR,
    OUTPUT_SUBDIRS, TIME_WINDOW_MINUTES,
    LABEL_FONT_SIZE, TICK_FONT_SIZE,
    PRODUCT_CMAPS,
)
from .utils import (
    load_events, load_flare_catalog, find_flare_row,
    get_flare_time_window, load_hdf5_map,
    find_nearest_map_time, save_figure, logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

LAT_BINS = np.arange(-90, 95, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hovmöller diagram of ROTI (time vs latitude)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--flares-csv", type=Path, default=DEFAULT_FLARES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flare-classes", nargs="+", default=list(FLARE_CLASSES), choices=FLARE_CLASSES)
    parser.add_argument("--window-minutes", type=float, default=TIME_WINDOW_MINUTES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def compute_lat_profile(points: np.ndarray) -> np.ndarray:
    if points.dtype.names is None:
        return np.full(len(LAT_BINS) - 1, np.nan)
    lats = points["lat"]
    vals = points["vals"]
    profile = np.full(len(LAT_BINS) - 1, np.nan)
    for i in range(len(LAT_BINS) - 1):
        lo, hi = LAT_BINS[i], LAT_BINS[i + 1]
        mask = (lats >= lo) & (lats < hi)
        if mask.any():
            profile[i] = float(np.nanmedian(vals[mask]))
    return profile


def plot_for_event(
    event: dict, results_dir: Path, catalog: pd.DataFrame,
    output_dir: Path, window_minutes: float, no_plots: bool,
) -> bool:
    flare_row = find_flare_row(event, catalog)
    if flare_row is None:
        return False
    peak_time = flare_row.get("peak_time")
    if pd.isna(peak_time):
        return False

    time_window = get_flare_time_window(peak_time, window_minutes)
    event_name = event.get("name", "unknown")

    timestamps, map_data = load_hdf5_map(event, results_dir, "roti", time_window)
    if not timestamps:
        return False

    if no_plots:
        logger.info(f"[{event_name}] {len(timestamps)} map times")
        return True

    profiles = []
    for t in timestamps:
        pts = map_data.get(t)
        if pts is not None:
            profiles.append(compute_lat_profile(pts))
        else:
            profiles.append(np.full(len(LAT_BINS) - 1, np.nan))

    Z = np.ma.array(profiles, mask=np.isnan(profiles))
    if Z.mask.all():
        logger.warning(f"[{event_name}] All NaN profiles")
        return False

    LAT_CENTERS = (LAT_BINS[:-1] + LAT_BINS[1:]) / 2

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SINGLE)
    T = np.array([t.timestamp() for t in timestamps])
    pcm = ax.pcolormesh(T, LAT_CENTERS, Z.T, shading="auto",
                        cmap=PRODUCT_CMAPS.get("roti", "viridis"), vmin=0, vmax=0.5)
    cbar = fig.colorbar(pcm, ax=ax, label="ROTI (TECu/min)")

    ax.set_ylabel("Latitude (deg)", fontsize=LABEL_FONT_SIZE)
    ax.set_xlabel("Time (UTC)", fontsize=LABEL_FONT_SIZE)
    ax.axvline(peak_time.timestamp(), color="red", linestyle="--", linewidth=1.5, label="Peak")
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=TICK_FONT_SIZE)

    def fmt_time(x, _):
        return pd.Timestamp(x, unit="s", tz="UTC").strftime("%H:%M")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_time))
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    filename = f"hovmoller_roti_{peak_time:%H-%M-%S_UTC}.png"
    save_figure(fig, event_name, OUTPUT_SUBDIRS["maps"], filename, output_dir)
    logger.info(f"[{event_name}] Saved Hovmöller diagram")
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

    events_to_process = events[:args.max_events] if args.max_events else events
    total = success = 0
    for event in events_to_process:
        if event.get("class") not in args.flare_classes:
            continue
        ok = plot_for_event(event, results_dir, catalog, output_dir,
                            args.window_minutes, args.no_plots)
        total += 1
        success += ok
        if not ok:
            logger.warning(f"[{event.get('name')}] Failed")
    logger.info(f"Done. {success}/{total} successful")


if __name__ == "__main__":
    main()
