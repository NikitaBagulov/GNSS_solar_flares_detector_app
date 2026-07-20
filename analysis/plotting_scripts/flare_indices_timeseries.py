#!/usr/bin/env python3
"""
Plot GNSS indices (Day/Night, GSFLAI, ISFAI) timeseries per product around flare.
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

from config import (
    PRODUCTS, PRODUCT_LABELS, FLARE_CLASSES, PLOT_FIGSIZE_SINGLE, PLOT_DPI,
    DEFAULT_RESULTS_DIR, DEFAULT_FLARES_CSV, DEFAULT_OUTPUT_DIR,
    OUTPUT_SUBDIRS, TIME_WINDOW_MINUTES, INDEX_COLUMNS,
)
from utils import (
    load_events, load_flare_catalog, event_file_path,
    normalize_time_column, get_flare_time_window, find_flare_row,
    load_indices_data, add_flare_markers, format_time_axis, save_figure,
    logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GNSS indices timeseries per product around flare")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--flares-csv", type=Path, default=DEFAULT_FLARES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flare-classes", nargs="+", default=list(FLARE_CLASSES), choices=FLARE_CLASSES)
    parser.add_argument("--window-minutes", type=float, default=TIME_WINDOW_MINUTES)
    parser.add_argument("--products", nargs="+", default=list(PRODUCTS), choices=PRODUCTS)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def plot_indices_for_event(
    event: dict,
    results_dir: Path,
    catalog: pd.DataFrame,
    output_dir: Path,
    window_minutes: float,
    products: list,
    no_plots: bool,
) -> dict[str, bool]:
    flare_row = find_flare_row(event, catalog)
    if flare_row is None:
        logger.warning(f"[{event.get('name')}] No matching flare in catalog")
        return {p: False for p in products}

    peak_time = flare_row.get("peak_time")
    if pd.isna(peak_time):
        logger.warning(f"[{event.get('name')}] No peak time")
        return {p: False for p in products}

    time_window = get_flare_time_window(peak_time, window_minutes)
    event_name = event.get("name", "unknown")
    flare_class = str(flare_row.get("class", event.get("class", "?"))).upper()[0]
    start_time = flare_row.get("start_time")
    end_time = flare_row.get("end_time")

    results = {}
    for product in products:
        if not event.get("indices", {}).get(product):
            logger.warning(f"[{event_name}] No indices for {product}")
            results[product] = False
            continue

        df = load_indices_data(event, results_dir, product, time_window)
        if df.empty:
            logger.warning(f"[{event_name}] Empty indices data for {product}")
            results[product] = False
            continue

        if no_plots:
            logger.info(f"[{event_name}] {product}: {len(df)} index points in window")
            results[product] = True
            continue

        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_SINGLE)
        ax2 = ax.twinx()
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))
        ax3.spines["right"].set_visible(True)

        axes = [ax, ax2, ax3]
        colors = ["tab:blue", "tab:green", "tab:red"]

        lines = []
        labels = []

        for axis, idx_col, color in zip(axes, INDEX_COLUMNS, colors):
            if idx_col not in df.columns:
                continue
            times = df["time"]
            values = pd.to_numeric(df[idx_col], errors="coerce")
            valid = ~values.isna()
            if not valid.any():
                continue
            line, = axis.plot(times[valid], values[valid], label=idx_col, color=color, linewidth=1.5)
            axis.set_ylabel(idx_col, color=color, fontsize=10)
            axis.tick_params(axis="y", colors=color, labelsize=9)
            axis.spines["left" if axis is ax else "right"].set_color(color)
            lines.append(line)
            labels.append(idx_col)

        if lines:
            ax.legend(lines, labels, loc="upper left", fontsize=9)

        add_flare_markers(ax, start_time, peak_time, end_time)
        format_time_axis(ax)
        ax.set_xlabel("Time (UTC)")
        ax.set_title(f"{event_name} | {flare_class}-class Flare | {PRODUCT_LABELS[product]} Indices", fontsize=12)

        fig.tight_layout()

        filename = f"indices_{product}_{peak_time:%H-%M-%S_UTC}.png"
        save_figure(fig, event_name, OUTPUT_SUBDIRS["timeseries_indices"], filename, output_dir)
        logger.info(f"[{event_name}] Saved {product} indices plot")
        results[product] = True

    return results


def main() -> None:
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    results_dir = args.results_dir.resolve()
    flares_csv = args.flares_csv.resolve()
    output_dir = args.output_dir.resolve()

    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        sys.exit(1)

    if not flares_csv.exists():
        logger.error(f"Flares catalog does not exist: {flares_csv}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading events from {results_dir}")
    events = load_events(results_dir)
    logger.info(f"Found {len(events)} events")

    logger.info(f"Loading flare catalog from {flares_csv}")
    catalog = load_flare_catalog(flares_csv)
    logger.info(f"Catalog has {len(catalog)} flares")

    events_to_process = events
    if args.max_events:
        events_to_process = events[:args.max_events]
        logger.info(f"Limited to first {args.max_events} events")

    total_processed = 0
    total_success = 0

    for event in events_to_process:
        event_name = event.get("name", "unknown")
        flare_class = event.get("class", "?")

        if flare_class not in args.flare_classes:
            continue

        results = plot_indices_for_event(
            event, results_dir, catalog, output_dir,
            args.window_minutes, args.products, args.no_plots
        )
        total_processed += 1
        success_count = sum(1 for v in results.values() if v)
        total_success += success_count

        if success_count == 0:
            logger.warning(f"[{event_name}] No product indices plotted")
        else:
            logger.info(f"[{event_name}] Success: {success_count}/{len(args.products)} products")

    logger.info(f"Done. Processed {total_processed} events, {total_success} successful product plots")


if __name__ == "__main__":
    main()