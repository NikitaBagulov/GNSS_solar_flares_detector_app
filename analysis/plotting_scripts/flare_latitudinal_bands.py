#!/usr/bin/env python3
"""
ROTI timeseries per latitudinal band for a flare event.
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
    LABEL_FONT_SIZE, TICK_FONT_SIZE, LEGEND_FONT_SIZE, TITLE_FONT_SIZE,
    LINE_WIDTH, COLORS_CONTRAST,
)
from .utils import (
    load_events, load_flare_catalog, find_flare_row,
    get_flare_time_window, load_hdf5_map,
    add_flare_markers, format_time_axis, apply_grid,
    save_figure, logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BANDS = [
    (-90, -60, "60S-90S"),
    (-60, -30, "30S-60S"),
    (-30, 0, "Eq-30S"),
    (0, 30, "Eq-30N"),
    (30, 60, "30N-60N"),
    (60, 90, "60N-90N"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROTI per latitudinal band")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--flares-csv", type=Path, default=DEFAULT_FLARES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flare-classes", nargs="+", default=list(FLARE_CLASSES), choices=FLARE_CLASSES)
    parser.add_argument("--window-minutes", type=float, default=TIME_WINDOW_MINUTES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


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
    start_time = flare_row.get("start_time")
    end_time = flare_row.get("end_time")

    tw = get_flare_time_window(peak_time, window_minutes)
    event_name = event.get("name", "unknown")

    timestamps, map_data = load_hdf5_map(event, results_dir, "roti", tw)
    if not timestamps:
        return False

    if no_plots:
        logger.info(f"[{event_name}] {len(timestamps)} map times")
        return True

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SINGLE)

    for idx, (lo, hi, label) in enumerate(BANDS):
        color = COLORS_CONTRAST[idx % len(COLORS_CONTRAST)]
        ts = []
        vals = []
        for t in timestamps:
            pts = map_data.get(t)
            if pts is None or pts.size == 0 or pts.dtype.names is None:
                continue
            mask = (pts["lat"] >= lo) & (pts["lat"] < hi)
            if mask.any():
                ts.append(t)
                vals.append(float(np.nanmedian(pts["vals"][mask])))
        if ts:
            ax.plot(ts, vals, color=color, linewidth=LINE_WIDTH, label=label)

    add_flare_markers(ax, start_time, peak_time, end_time)
    apply_grid(ax)
    format_time_axis(ax)
    ax.set_xlabel("Time (UTC)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Median ROTI (TECu/min)", fontsize=LABEL_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE, ncol=2)
    ax.tick_params(labelsize=TICK_FONT_SIZE)

    fig.tight_layout()
    filename = f"roti_lat_bands_{peak_time:%H-%M-%S_UTC}.png"
    save_figure(fig, event_name, OUTPUT_SUBDIRS["maps"], filename, output_dir)
    logger.info(f"[{event_name}] Saved latitudinal bands plot")
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
