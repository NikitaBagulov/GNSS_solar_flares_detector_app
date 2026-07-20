#!/usr/bin/env python3
"""
Flare position on solar disk vs ROTI response amplitude.
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
    FLARE_CLASSES, FLARE_CLASS_MARKERS, FLARE_CLASS_COLORS,
    SOLAR_RADIUS_ARCSEC, PLOT_FIGSIZE_SINGLE, PLOT_FIGSIZE_STATS, PLOT_DPI,
    DEFAULT_RESULTS_DIR, DEFAULT_FLARES_CSV, DEFAULT_OUTPUT_DIR,
    OUTPUT_SUBDIRS, TIME_WINDOW_MINUTES,
    LABEL_FONT_SIZE, TICK_FONT_SIZE, LEGEND_FONT_SIZE,
    LINE_WIDTH_THICK,
)
from .utils import (
    load_events, load_flare_catalog, find_flare_row,
    get_flare_time_window, load_hdf5_map,
    find_nearest_map_time, apply_grid, save_figure, logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flare position on disk vs ROTI response")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--flares-csv", type=Path, default=DEFAULT_FLARES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flare-classes", nargs="+", default=list(FLARE_CLASSES), choices=FLARE_CLASSES)
    parser.add_argument("--window-minutes", type=float, default=TIME_WINDOW_MINUTES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


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
    stats_dir = output_dir / OUTPUT_SUBDIRS["statistics"]
    stats_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(results_dir)
    catalog = load_flare_catalog(flares_csv)

    events_to_process = events[:args.max_events] if args.max_events else events

    rows = []
    for event in events_to_process:
        fc = event.get("class", "?")
        if fc not in args.flare_classes:
            continue
        flare_row = find_flare_row(event, catalog)
        if flare_row is None:
            continue

        hpc_x = flare_row.get("hpc_x")
        hpc_y = flare_row.get("hpc_y")
        if pd.isna(hpc_x) or pd.isna(hpc_y):
            continue
        hpc_x, hpc_y = float(hpc_x), float(hpc_y)

        peak_time = flare_row.get("peak_time")
        if pd.isna(peak_time):
            continue

        tw = get_flare_time_window(peak_time, args.window_minutes)
        timestamps, map_data = load_hdf5_map(event, results_dir, "roti", tw)
        if not timestamps:
            continue
        nearest = find_nearest_map_time(timestamps, peak_time)
        if nearest is None:
            continue
        pts = map_data.get(nearest)
        if pts is None or pts.size == 0 or pts.dtype.names is None:
            continue

        roti_max = float(np.nanmax(pts["vals"]))
        dist = np.sqrt(hpc_x ** 2 + hpc_y ** 2) / SOLAR_RADIUS_ARCSEC

        rows.append({
            "event": event.get("name"),
            "flare_class": fc,
            "hpc_x": hpc_x,
            "hpc_y": hpc_y,
            "dist_norm": dist,
            "roti_max": roti_max,
            "peak_flux": flare_row.get("peak_flux"),
        })

    if not rows:
        logger.warning("No data collected")
        return

    df = pd.DataFrame(rows)
    if args.no_plots:
        csv_path = stats_dir / "position_response.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {csv_path}")
        return

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121)
    disk = plt.Circle((0, 0), SOLAR_RADIUS_ARCSEC, color="black", fill=False, linewidth=LINE_WIDTH_THICK, alpha=0.7)
    ax1.add_patch(disk)
    for fc in FLARE_CLASSES:
        subset = df[df["flare_class"] == fc]
        if subset.empty:
            continue
        sizes = 20 + 80 * (subset["roti_max"] / subset["roti_max"].max())
        ax1.scatter(subset["hpc_x"], subset["hpc_y"], s=sizes,
                   alpha=0.7, marker=FLARE_CLASS_MARKERS[fc],
                   color=FLARE_CLASS_COLORS[fc], label=f"{fc}-class",
                   edgecolors="black", linewidths=0.3)
    ax1.set_aspect("equal")
    ax1.set_xlabel("HPC X (arcsec)", fontsize=LABEL_FONT_SIZE)
    ax1.set_ylabel("HPC Y (arcsec)", fontsize=LABEL_FONT_SIZE)
    ax1.legend(fontsize=LEGEND_FONT_SIZE)
    apply_grid(ax1)
    ax1.tick_params(labelsize=TICK_FONT_SIZE)

    ax2 = fig.add_subplot(122)
    for fc in FLARE_CLASSES:
        subset = df[df["flare_class"] == fc]
        if subset.empty:
            continue
        ax2.scatter(subset["dist_norm"], subset["roti_max"],
                   s=60, alpha=0.7, marker=FLARE_CLASS_MARKERS[fc],
                   color=FLARE_CLASS_COLORS[fc], label=f"{fc}-class",
                   edgecolors="black", linewidths=0.3)
    ax2.set_xlabel("Normalized distance from disk center", fontsize=LABEL_FONT_SIZE)
    ax2.set_ylabel("Max ROTI (TECu/min)", fontsize=LABEL_FONT_SIZE)
    ax2.legend(fontsize=LEGEND_FONT_SIZE)
    apply_grid(ax2)
    ax2.tick_params(labelsize=TICK_FONT_SIZE)

    fig.tight_layout()
    fig.savefig(stats_dir / "position_response.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved position response plot to {stats_dir}")


if __name__ == "__main__":
    main()
