#!/usr/bin/env python3
"""
Estimate ROTI perturbation lifetime vs flare class.
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
    LINE_WIDTH,
)
from .utils import (
    load_events, load_flare_catalog, find_flare_row,
    get_flare_time_window, load_hdf5_map,
    find_nearest_map_time, apply_grid, save_figure, logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROTI perturbation lifetime vs flare class")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--flares-csv", type=Path, default=DEFAULT_FLARES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flare-classes", nargs="+", default=list(FLARE_CLASSES), choices=FLARE_CLASSES)
    parser.add_argument("--window-minutes", type=float, default=TIME_WINDOW_MINUTES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def estimate_fwhm(timestamps: list, roti_means: list) -> float:
    if len(roti_means) < 3:
        return np.nan
    vals = np.array(roti_means)
    baseline = np.percentile(vals, 10)
    peak = np.max(vals)
    half = baseline + (peak - baseline) / 2

    above = vals >= half
    if above.sum() < 2:
        return np.nan

    indices = np.where(above)[0]
    start = timestamps[indices[0]]
    end = timestamps[indices[-1]]
    return (end - start).total_seconds() / 60.0


def mean_roti_series(timestamps, map_data):
    means = []
    for t in timestamps:
        pts = map_data.get(t)
        if pts is not None and pts.size and pts.dtype.names is not None:
            means.append(float(np.nanmedian(pts["vals"])))
        else:
            means.append(np.nan)
    return means


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

        means = mean_roti_series(timestamps, map_data)
        if len([m for m in means if not np.isnan(m)]) < 3:
            continue

        fwhm = estimate_fwhm(timestamps, means)
        nearest = find_nearest_map_time(timestamps, peak_time)
        pts = map_data.get(nearest) if nearest else None
        roti_peak = float(np.nanmax(pts["vals"])) if pts is not None and pts.dtype.names is not None else np.nan

        rows.append({
            "event": event.get("name"),
            "flare_class": fc,
            "fwhm_min": fwhm,
            "roti_peak": roti_peak,
            "peak_flux": flare_row.get("peak_flux"),
        })

    if not rows:
        logger.warning("No data collected")
        return

    df = pd.DataFrame(rows)
    if args.no_plots:
        csv_path = stats_dir / "roti_lifetime.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {csv_path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, xcol, xlabel, logx in [
        (axes[0], "peak_flux", "GOES Peak Flux (W m$^{-2}$)", True),
        (axes[1], "roti_peak", "Peak ROTI (TECu/min)", False),
    ]:
        valid = df[xcol].notna() & df["fwhm_min"].notna()
        if not valid.any():
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        for fc in FLARE_CLASSES:
            subset = df[valid & (df["flare_class"] == fc)]
            if subset.empty:
                continue
            ax.scatter(subset[xcol], subset["fwhm_min"],
                      s=60, alpha=0.7, marker=FLARE_CLASS_MARKERS[fc],
                      color=FLARE_CLASS_COLORS[fc], label=f"{fc}-class",
                      edgecolors="black", linewidths=0.3)

        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel("FWHM (min)", fontsize=LABEL_FONT_SIZE)
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        apply_grid(ax)
        ax.tick_params(labelsize=TICK_FONT_SIZE)

    fig.tight_layout()
    fig.savefig(stats_dir / "roti_lifetime.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=PLOT_FIGSIZE_STATS)
    for fc in FLARE_CLASSES:
        vals = df[df["flare_class"] == fc]["fwhm_min"].dropna()
        if vals.empty:
            continue
        ax.hist(vals, bins=15, alpha=0.5, color=FLARE_CLASS_COLORS[fc],
               label=f"{fc}-class", density=True)
    ax.set_xlabel("FWHM (min)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Density", fontsize=LABEL_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    apply_grid(ax)
    ax.tick_params(labelsize=TICK_FONT_SIZE)
    fig2.tight_layout()
    fig2.savefig(stats_dir / "roti_lifetime_hist.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig2)

    logger.info(f"Saved lifetime plots to {stats_dir}")


if __name__ == "__main__":
    main()
