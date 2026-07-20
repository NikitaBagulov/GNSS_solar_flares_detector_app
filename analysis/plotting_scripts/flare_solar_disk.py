#!/usr/bin/env python3
"""
Plot flare position on solar disk using HPC coordinates from catalog.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    FLARE_CLASSES, FLARE_CLASS_MARKERS, FLARE_CLASS_COLORS,
    SOLAR_RADIUS_ARCSEC, PLOT_FIGSIZE_SINGLE, PLOT_DPI,
    DEFAULT_RESULTS_DIR, DEFAULT_FLARES_CSV, DEFAULT_OUTPUT_DIR,
    OUTPUT_SUBDIRS, LABEL_FONT_SIZE, TICK_FONT_SIZE, LEGEND_FONT_SIZE, TITLE_FONT_SIZE,
)
from .utils import (
    load_events, load_flare_catalog, event_file_path,
    load_solar_image, plot_solar_disk_base, find_flare_row,
    convert_hpc_to_pixel, save_figure,
    logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot flare position on solar disk")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--flares-csv", type=Path, default=DEFAULT_FLARES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flare-classes", nargs="+", default=list(FLARE_CLASSES), choices=FLARE_CLASSES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def plot_solar_disk_for_event(
    event: dict,
    results_dir: Path,
    catalog: pd.DataFrame,
    output_dir: Path,
    no_plots: bool,
) -> bool:
    flare_row = find_flare_row(event, catalog)
    if flare_row is None:
        logger.warning(f"[{event.get('name')}] No matching flare in catalog")
        return False

    hpc_x = flare_row.get("hpc_x")
    hpc_y = flare_row.get("hpc_y")
    if pd.isna(hpc_x) or pd.isna(hpc_y):
        logger.warning(f"[{event.get('name')}] Missing HPC coordinates")
        return False

    solar_image = load_solar_image(event, results_dir)
    event_name = event.get("name", "unknown")
    flare_class = str(flare_row.get("class", event.get("class", "?"))).upper()[0]
    peak_time = flare_row.get("peak_time")

    if no_plots:
        logger.info(f"[{event_name}] Flare {flare_class} at HPC=({hpc_x:.0f}, {hpc_y:.0f}) arcsec, peak={peak_time}")
        return True

    fig = plt.figure(figsize=PLOT_FIGSIZE_SINGLE)
    ax = fig.add_subplot(1, 1, 1)

    plot_solar_disk_base(
        ax,
        flare_hpc_x=float(hpc_x),
        flare_hpc_y=float(hpc_y),
        solar_image=solar_image,
    )

    marker = FLARE_CLASS_MARKERS.get(flare_class, "o")
    color = FLARE_CLASS_COLORS.get(flare_class, "red")

    if solar_image is not None:
        x_px, y_px, r_px = convert_hpc_to_pixel(solar_image, float(hpc_x), float(hpc_y))
        if x_px is not None:
            ax.scatter([x_px], [y_px], s=200, color=color, marker=marker, edgecolor="white", linewidth=1.5, zorder=15)
            ax.annotate(
                f"{flare_class}",
                (x_px, y_px), xytext=(15, -15), textcoords="offset points",
                color="white", fontsize=LEGEND_FONT_SIZE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7),
            )

    fig.tight_layout()

    filename = f"solar_disk_{flare_class}_{event_name}.png"
    save_figure(fig, event_name, OUTPUT_SUBDIRS["solar_disk"], filename, output_dir)
    logger.info(f"[{event_name}] Saved solar disk plot with flare at HPC=({hpc_x:.0f}, {hpc_y:.0f})")
    return True


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

        success = plot_solar_disk_for_event(event, results_dir, catalog, output_dir, args.no_plots)
        total_processed += 1
        if success:
            total_success += 1
        else:
            logger.warning(f"[{event_name}] Failed to plot solar disk")

    logger.info(f"Done. Processed {total_processed} events, {total_success} successful plots")


if __name__ == "__main__":
    main()
