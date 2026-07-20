#!/usr/bin/env python3
"""
Mean ROTI + GOES X-ray timeseries on twin axes.
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
    LINE_WIDTH, LINE_WIDTH_THICK, XRAY_COLORS,
    LABEL_FONT_SIZE, TICK_FONT_SIZE, LEGEND_FONT_SIZE,
)
from .utils import (
    load_events, load_flare_catalog, find_flare_row,
    get_flare_time_window, load_hdf5_map, load_goes_xray,
    find_nearest_map_time, add_flare_markers, format_time_axis, apply_grid,
    save_figure, logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mean ROTI vs GOES X-ray timeseries")
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

    time_window = get_flare_time_window(peak_time, window_minutes)
    event_name = event.get("name", "unknown")
    start_time = flare_row.get("start_time")
    end_time = flare_row.get("end_time")
    flare_class = str(flare_row.get("class", "?")).upper()[0]

    timestamps, map_data = load_hdf5_map(event, results_dir, "roti", time_window)
    if not timestamps:
        return False

    goes_df = load_goes_xray(event, results_dir, time_window)

    if no_plots:
        logger.info(f"[{event_name}] {len(timestamps)} maps, {len(goes_df)} GOES points")
        return True

    roti_means = []
    roti_times = []
    for t in timestamps:
        pts = map_data.get(t)
        if pts is None or pts.size == 0 or pts.dtype.names is None:
            continue
        vals = pts["vals"]
        roti_means.append(float(np.nanmedian(vals)))
        roti_times.append(t)

    fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE_SINGLE)
    ax2 = ax1.twinx()

    if roti_times:
        ax1.plot(roti_times, roti_means, color="black", linewidth=LINE_WIDTH, label="Median ROTI")
        ax1.set_ylabel("Median ROTI (TECu/min)", fontsize=LABEL_FONT_SIZE, color="black")
        ax1.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

    if not goes_df.empty and "xrsb" in goes_df.columns:
        ax2.semilogy(goes_df["time"], goes_df["xrsb"],
                     color=XRAY_COLORS.get("xrsb", "red"), linewidth=LINE_WIDTH,
                     label="GOES XRS-B")
        ax2.set_ylabel("GOES XRS-B (W m$^{-2}$)", fontsize=LABEL_FONT_SIZE,
                       color=XRAY_COLORS.get("xrsb", "red"))
        ax2.tick_params(axis="y", labelsize=TICK_FONT_SIZE,
                        colors=XRAY_COLORS.get("xrsb", "red"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=LEGEND_FONT_SIZE)

    add_flare_markers(ax1, start_time, peak_time, end_time)
    apply_grid(ax1)
    format_time_axis(ax1)
    ax1.set_xlabel("Time (UTC)", fontsize=LABEL_FONT_SIZE)
    ax1.tick_params(labelsize=TICK_FONT_SIZE)

    fig.tight_layout()
    filename = f"mean_roti_xray_{peak_time:%H-%M-%S_UTC}.png"
    save_figure(fig, event_name, OUTPUT_SUBDIRS["timeseries_xray_euv"], filename, output_dir)
    logger.info(f"[{event_name}] Saved mean ROTI + X-ray plot")
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
