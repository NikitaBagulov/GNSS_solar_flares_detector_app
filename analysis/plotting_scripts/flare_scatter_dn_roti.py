#!/usr/bin/env python3
"""
Scatter: Day/Night index vs median ROTI per event, colored by flare class.
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
    PLOT_FIGSIZE_STATS, PLOT_DPI,
    DEFAULT_RESULTS_DIR, DEFAULT_FLARES_CSV, DEFAULT_OUTPUT_DIR,
    OUTPUT_SUBDIRS, TIME_WINDOW_MINUTES,
    LABEL_FONT_SIZE, TICK_FONT_SIZE, LEGEND_FONT_SIZE,
    COLORS_CONTRAST,
)
from .utils import (
    load_events, load_flare_catalog, find_flare_row,
    get_flare_time_window, load_hdf5_map, load_indices_csv,
    find_nearest_map_time, apply_grid, save_figure, logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day/Night vs ROTI scatter")
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
        flare_class = event.get("class", "?")
        if flare_class not in args.flare_classes:
            continue

        flare_row = find_flare_row(event, catalog)
        if flare_row is None:
            continue
        peak_time = flare_row.get("peak_time")
        if pd.isna(peak_time):
            continue

        time_window = get_flare_time_window(peak_time, args.window_minutes)
        timestamps, map_data = load_hdf5_map(event, results_dir, "roti", time_window)
        if not timestamps:
            continue

        nearest = find_nearest_map_time(timestamps, peak_time)
        if nearest is None:
            continue

        pts = map_data.get(nearest)
        if pts is None or pts.size == 0 or pts.dtype.names is None:
            continue

        roti_median = float(np.nanmedian(pts["vals"]))

        idx_df = load_indices_csv(event, results_dir, "roti",
                                  get_flare_time_window(nearest, 5.0))
        dn = None
        if not idx_df.empty and "day_night" in idx_df.columns:
            vals = pd.to_numeric(idx_df["day_night"], errors="coerce").dropna()
            if not vals.empty:
                dn = float(vals.iloc[0])

        rows.append({
            "event": event.get("name"),
            "flare_class": flare_class,
            "roti_median": roti_median,
            "day_night": dn,
            "peak_flux": flare_row.get("peak_flux"),
        })

    if not rows:
        logger.warning("No data collected")
        return

    df = pd.DataFrame(rows)
    if args.no_plots:
        csv_path = stats_dir / "dn_vs_roti.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {csv_path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, xcol, xlabel in [
        (axes[0], "day_night", "Day/Night Index"),
        (axes[1], "peak_flux", "GOES Peak Flux (W m$^{-2}$)"),
    ]:
        valid = df[xcol].notna()
        if not valid.any():
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        for fc in FLARE_CLASSES:
            subset = df[valid & (df["flare_class"] == fc)]
            if subset.empty:
                continue
            ax.scatter(subset[xcol], subset["roti_median"],
                      s=60, alpha=0.7, marker=FLARE_CLASS_MARKERS[fc],
                      color=FLARE_CLASS_COLORS[fc], label=f"{fc}-class",
                      edgecolors="black", linewidths=0.3)
        if xcol == "peak_flux":
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("Median ROTI (TECu/min)", fontsize=LABEL_FONT_SIZE)
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        apply_grid(ax)
        ax.tick_params(labelsize=TICK_FONT_SIZE)

    fig.tight_layout()
    fig.savefig(stats_dir / "dn_vs_roti_scatter.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved scatter plots to {stats_dir}")


if __name__ == "__main__":
    main()
