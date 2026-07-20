#!/usr/bin/env python3
"""
Plot GOES X-ray and SOHO SEM EUV timeseries around flare peak.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from .config import (
    FLARE_CLASSES, PLOT_FIGSIZE_SINGLE, PLOT_DPI,
    DEFAULT_RESULTS_DIR, DEFAULT_FLARES_CSV, DEFAULT_OUTPUT_DIR,
    OUTPUT_SUBDIRS, GOES_XRAY_COLUMNS, SOHO_SEM_COLUMNS,
    TIME_WINDOW_MINUTES,
    LINE_WIDTH, XRAY_COLORS, EUV_COLOR,
    LABEL_FONT_SIZE, TICK_FONT_SIZE, LEGEND_FONT_SIZE, TITLE_FONT_SIZE,
)
from .utils import (
    load_events, load_flare_catalog, event_file_path,
    normalize_time_column, get_flare_time_window, find_flare_row,
    add_flare_markers, format_time_axis, apply_grid, save_figure,
    load_goes_xray, load_soho_sem,
    logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GOES X-ray and SOHO SEM EUV timeseries around flare")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--flares-csv", type=Path, default=DEFAULT_FLARES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flare-classes", nargs="+", default=list(FLARE_CLASSES), choices=FLARE_CLASSES)
    parser.add_argument("--window-minutes", type=float, default=TIME_WINDOW_MINUTES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def plot_xray_euv_timeseries(
    event: dict,
    results_dir: Path,
    catalog: pd.DataFrame,
    output_dir: Path,
    window_minutes: float,
    no_plots: bool,
) -> bool:
    flare_row = find_flare_row(event, catalog)
    if flare_row is None:
        logger.warning(f"[{event.get('name')}] No matching flare in catalog")
        return False

    peak_time = flare_row.get("peak_time")
    if pd.isna(peak_time):
        logger.warning(f"[{event.get('name')}] No peak time in catalog")
        return False

    time_window = get_flare_time_window(peak_time, window_minutes)

    goes_df = load_goes_xray(event, results_dir, time_window)
    sem_df = load_soho_sem(event, results_dir, time_window)

    if goes_df.empty and sem_df.empty:
        logger.warning(f"[{event.get('name')}] No GOES or SOHO data in window")
        return False

    event_name = event.get("name", "unknown")
    flare_class = str(flare_row.get("class", event.get("class", "?"))).upper()[0]
    start_time = flare_row.get("start_time")
    end_time = flare_row.get("end_time")

    if no_plots:
        logger.info(f"[{event_name}] GOES points: {len(goes_df)}, SOHO points: {len(sem_df)}")
        return True

    fig, axes = plt.subplots(2, 1, figsize=PLOT_FIGSIZE_SINGLE, sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    if not goes_df.empty:
        ax = axes[0]
        for col in GOES_XRAY_COLUMNS:
            if col in goes_df.columns:
                label = f"GOES {col.upper()}"
                color = XRAY_COLORS.get(col, "black")
                ax.semilogy(goes_df["time"], goes_df[col], label=label,
                           color=color, linewidth=LINE_WIDTH)
        ax.set_ylabel("Flux (W m$^{-2}$)", fontsize=LABEL_FONT_SIZE)
        ax.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE)
        apply_grid(ax)
        add_flare_markers(ax, start_time, peak_time, end_time)
        ax.tick_params(labelsize=TICK_FONT_SIZE)

    if not sem_df.empty:
        ax = axes[1]
        for col in SOHO_SEM_COLUMNS:
            if col in sem_df.columns:
                label_map = {"flux_26_34": "26-34 nm", "flux_01_50": "0.1-50 nm"}
                label = f"SOHO SEM {label_map.get(col, col)}"
                ax.plot(sem_df["time"], sem_df[col], label=label,
                       color=EUV_COLOR, linewidth=LINE_WIDTH)
        ax.set_ylabel("Flux (phot. cm$^{-2}$ s$^{-1}$)", fontsize=LABEL_FONT_SIZE)
        ax.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE)
        apply_grid(ax)
        add_flare_markers(ax, start_time, peak_time, end_time)
        ax.tick_params(labelsize=TICK_FONT_SIZE)

    format_time_axis(axes[-1])
    axes[-1].tick_params(labelsize=TICK_FONT_SIZE)

    fig.tight_layout()

    filename = f"xray_euv_{peak_time:%H-%M-%S_UTC}.png"
    save_figure(fig, event_name, OUTPUT_SUBDIRS["timeseries_xray_euv"], filename, output_dir)
    logger.info(f"[{event_name}] Saved X-ray/EUV timeseries")
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

        success = plot_xray_euv_timeseries(
            event, results_dir, catalog, output_dir,
            args.window_minutes, args.no_plots
        )
        total_processed += 1
        if success:
            total_success += 1
        else:
            logger.warning(f"[{event_name}] Failed to plot X-ray/EUV timeseries")

    logger.info(f"Done. Processed {total_processed} events, {total_success} successful plots")


if __name__ == "__main__":
    main()
