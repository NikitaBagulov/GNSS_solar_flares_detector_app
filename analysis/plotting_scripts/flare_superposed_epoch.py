#!/usr/bin/env python3
"""
Superposed epoch analysis: align ROTI, Day/Night, X-ray to peak time, average by class.
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
    FLARE_CLASSES, FLARE_CLASS_COLORS,
    PLOT_FIGSIZE_STATS, PLOT_DPI,
    DEFAULT_RESULTS_DIR, DEFAULT_FLARES_CSV, DEFAULT_OUTPUT_DIR,
    OUTPUT_SUBDIRS, TIME_WINDOW_MINUTES,
    LABEL_FONT_SIZE, TICK_FONT_SIZE, LEGEND_FONT_SIZE,
    LINE_WIDTH, XRAY_COLORS,
)
from .utils import (
    load_events, load_flare_catalog, find_flare_row,
    get_flare_time_window, load_hdf5_map, load_goes_xray, load_indices_csv,
    save_figure, logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EPOCH_WINDOW_MINUTES = 20
BIN_SECONDS = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Superposed epoch analysis")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--flares-csv", type=Path, default=DEFAULT_FLARES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flare-classes", nargs="+", default=list(FLARE_CLASSES), choices=FLARE_CLASSES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def epoch_bin(offset_seconds: float) -> float:
    return round(offset_seconds / BIN_SECONDS) * BIN_SECONDS / 60.0


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

    by_class = {fc: {"roti": [], "xrsb": [], "dn": []} for fc in FLARE_CLASSES}

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

        tw = get_flare_time_window(peak_time, EPOCH_WINDOW_MINUTES)

        timestamps, map_data = load_hdf5_map(event, results_dir, "roti", tw)
        for t in timestamps:
            pts = map_data.get(t)
            if pts is not None and pts.size and pts.dtype.names is not None:
                offset = epoch_bin((t - peak_time).total_seconds())
                by_class[fc]["roti"].append((offset, float(np.nanmedian(pts["vals"]))))

        goes = load_goes_xray(event, results_dir, tw)
        if not goes.empty and "xrsb" in goes.columns:
            for _, row in goes.iterrows():
                offset = epoch_bin((row["time"] - peak_time).total_seconds())
                by_class[fc]["xrsb"].append((offset, row["xrsb"]))

        idx_df = load_indices_csv(event, results_dir, "roti", tw)
        if not idx_df.empty and "day_night" in idx_df.columns:
            for _, row in idx_df.iterrows():
                offset = epoch_bin((row["time"] - peak_time).total_seconds())
                by_class[fc]["dn"].append((offset, row["day_night"]))

    if args.no_plots:
        logger.info("Epoch data collected, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, key, ylabel in [
        (axes[0], "roti", "Median ROTI (TECu/min)"),
        (axes[1], "xrsb", "GOES XRS-B (W m$^{-2}$)"),
        (axes[2], "dn", "Day/Night Index"),
    ]:
        for fc in FLARE_CLASSES:
            data = by_class[fc][key]
            if not data:
                continue
            df = pd.DataFrame(data, columns=["offset_min", "value"]).dropna()
            if df.empty:
                continue
            grouped = df.groupby("offset_min")["value"]
            means = grouped.mean()
            stds = grouped.std()
            counts = grouped.count()

            x = means.index.values
            ax.plot(x, means.values, color=FLARE_CLASS_COLORS[fc],
                    linewidth=LINE_WIDTH, label=f"{fc}-class (n={counts.sum():.0f})")
            ax.fill_between(x, means.values - stds.values, means.values + stds.values,
                           color=FLARE_CLASS_COLORS[fc], alpha=0.15)

        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Peak")
        ax.set_xlabel("Time from peak (min)", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=TICK_FONT_SIZE)

    fig.tight_layout()
    fig.savefig(stats_dir / "superposed_epoch.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved superposed epoch plot to {stats_dir}")


if __name__ == "__main__":
    main()
