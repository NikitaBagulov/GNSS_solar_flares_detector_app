#!/usr/bin/env python3
"""
Compare ROTI response on rising vs declining phase of GOES X-ray.
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
    get_flare_time_window, load_hdf5_map, load_goes_xray,
    find_nearest_map_time, apply_grid, save_figure, logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PHASE_WINDOW_MINUTES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROTI on rising vs declining X-ray phase")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--flares-csv", type=Path, default=DEFAULT_FLARES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flare-classes", nargs="+", default=list(FLARE_CLASSES), choices=FLARE_CLASSES)
    parser.add_argument("--window-minutes", type=float, default=TIME_WINDOW_MINUTES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def mean_roti_in_window(timestamps, map_data, t0, window_minutes):
    tw = get_flare_time_window(t0, window_minutes)
    vals = []
    for t in timestamps:
        if tw[0] <= t <= tw[1]:
            pts = map_data.get(t)
            if pts is not None and pts.size and pts.dtype.names is not None:
                vals.append(float(np.nanmedian(pts["vals"])))
    return np.mean(vals) if vals else np.nan


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
        peak_time = flare_row.get("peak_time")
        if pd.isna(peak_time):
            continue

        tw = get_flare_time_window(peak_time, args.window_minutes)
        timestamps, map_data = load_hdf5_map(event, results_dir, "roti", tw)
        if not timestamps:
            continue

        rising = mean_roti_in_window(timestamps, map_data,
                                     peak_time - pd.Timedelta(minutes=PHASE_WINDOW_MINUTES / 2),
                                     PHASE_WINDOW_MINUTES / 2)
        declining = mean_roti_in_window(timestamps, map_data,
                                        peak_time + pd.Timedelta(minutes=PHASE_WINDOW_MINUTES / 2),
                                        PHASE_WINDOW_MINUTES / 2)

        if np.isnan(rising) or np.isnan(declining):
            continue

        rows.append({
            "event": event.get("name"),
            "flare_class": fc,
            "rising_roti": rising,
            "declining_roti": declining,
            "ratio": declining / rising if rising > 0 else np.nan,
            "peak_flux": flare_row.get("peak_flux"),
        })

    if not rows:
        logger.warning("No data collected")
        return

    df = pd.DataFrame(rows)
    if args.no_plots:
        csv_path = stats_dir / "roti_phase.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {csv_path}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    panels = [
        (axes[0], "rising_roti", "declining_roti", "Rising ROTI (TECu/min)", "Declining ROTI (TECu/min)", False),
        (axes[1], "peak_flux", "rising_roti", "GOES Peak Flux (W m$^{-2}$)", "Rising ROTI (TECu/min)", True),
        (axes[2], "peak_flux", "declining_roti", "GOES Peak Flux (W m$^{-2}$)", "Declining ROTI (TECu/min)", True),
    ]
    for ax, xcol, ycol, xlabel, ylabel, logx in panels:
        valid = df[xcol].notna() & df[ycol].notna()
        for fc in FLARE_CLASSES:
            subset = df[valid & (df["flare_class"] == fc)]
            if subset.empty:
                continue
            ax.scatter(subset[xcol], subset[ycol],
                      s=60, alpha=0.7, marker=FLARE_CLASS_MARKERS[fc],
                      color=FLARE_CLASS_COLORS[fc], label=f"{fc}-class",
                      edgecolors="black", linewidths=0.3)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        apply_grid(ax)
        ax.tick_params(labelsize=TICK_FONT_SIZE)
        if xcol == "rising_roti" and ycol == "declining_roti":
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    fig.tight_layout()
    fig.savefig(stats_dir / "roti_phase_comparison.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=PLOT_FIGSIZE_STATS)
    for fc in FLARE_CLASSES:
        subset = df[df["flare_class"] == fc]["ratio"].dropna()
        if subset.empty:
            continue
        ax.hist(subset, bins=15, alpha=0.5, color=FLARE_CLASS_COLORS[fc],
               label=f"{fc}-class", density=True)
    ax.axvline(1, color="red", linestyle="--", linewidth=1.5, label="Equal")
    ax.set_xlabel("Declining / Rising ROTI ratio", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Density", fontsize=LABEL_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    apply_grid(ax)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    fig2.tight_layout()
    fig2.savefig(stats_dir / "roti_phase_ratio_hist.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig2)

    logger.info(f"Saved phase comparison plots to {stats_dir}")


if __name__ == "__main__":
    main()
