#!/usr/bin/env python3
"""
Combined multi-panel dashboard: 4 maps + 4 index panels + solar disk + X-ray/EUV.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd

from .config import (
    PRODUCTS, PRODUCT_LABELS, PRODUCT_VMIN_VMAX, PRODUCT_CMAPS,
    FLARE_CLASSES, PLOT_FIGSIZE_DASHBOARD, PLOT_DPI, MAP_POINT_SIZE, MAP_ALPHA,
    SOLAR_RADIUS_ARCSEC, TIME_WINDOW_MINUTES,
    DEFAULT_RESULTS_DIR, DEFAULT_FLARES_CSV, DEFAULT_OUTPUT_DIR,
    OUTPUT_SUBDIRS, GOES_XRAY_COLUMNS, SOHO_SEM_COLUMNS, INDEX_COLUMNS,
    LINE_WIDTH, INDEX_COLORS, XRAY_COLORS, EUV_COLOR,
    LABEL_FONT_SIZE, TICK_FONT_SIZE, LEGEND_FONT_SIZE, TITLE_FONT_SIZE,
)
from .utils import (
    load_events, load_flare_catalog, event_file_path,
    normalize_time_column, get_flare_time_window, find_flare_row,
    load_solar_image, plot_solar_disk_base, plot_global_map,
    add_flare_markers, format_time_axis, apply_grid, fill_negative,
    load_hdf5_map, load_indices_csv, load_goes_xray, load_soho_sem,
    find_nearest_map_time, save_figure, logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INDEX_LABELS = {
    "day_night": "Day/Night",
    "gsflai": "GSFLAI",
    "isfai": "ISFAI",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot combined dashboard for each flare")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--flares-csv", type=Path, default=DEFAULT_FLARES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flare-classes", nargs="+", default=list(FLARE_CLASSES), choices=FLARE_CLASSES)
    parser.add_argument("--window-minutes", type=float, default=TIME_WINDOW_MINUTES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def plot_dashboard_for_event(
    event: dict,
    results_dir: Path,
    catalog: pd.DataFrame,
    output_dir: Path,
    window_minutes: float,
    no_plots: bool,
) -> dict[str, bool]:
    flare_row = find_flare_row(event, catalog)
    if flare_row is None:
        logger.warning(f"[{event.get('name')}] No matching flare in catalog")
        return {"dashboard": False}

    peak_time = flare_row.get("peak_time")
    if pd.isna(peak_time):
        logger.warning(f"[{event.get('name')}] No peak time")
        return {"dashboard": False}

    time_window = get_flare_time_window(peak_time, window_minutes)
    event_name = event.get("name", "unknown")
    flare_class = str(flare_row.get("class", event.get("class", "?"))).upper()[0]
    start_time = flare_row.get("start_time")
    end_time = flare_row.get("end_time")

    map_timestamps = {}
    map_data = {}
    for product in PRODUCTS:
        if event.get("maps", {}).get(product):
            timestamps, data = load_hdf5_map(event, results_dir, product, time_window)
            if timestamps:
                map_timestamps[product] = timestamps
                map_data[product] = data

    all_timestamps = set()
    for ts in map_timestamps.values():
        all_timestamps.update(ts)
    all_timestamps = sorted(all_timestamps)

    if not all_timestamps:
        logger.warning(f"[{event_name}] No map timestamps in window")
        return {"dashboard": False}

    nearest_map_time = find_nearest_map_time(all_timestamps, peak_time, tolerance_minutes=window_minutes)
    if nearest_map_time is None:
        nearest_map_time = all_timestamps[0]
        logger.warning(f"[{event_name}] No map time near peak, using first: {nearest_map_time}")

    index_time_window = get_flare_time_window(nearest_map_time, window_minutes=5.0)
    indices_data = {}
    for product in PRODUCTS:
        df = load_indices_csv(event, results_dir, product, index_time_window)
        if not df.empty:
            indices_data[product] = df

    goes_df = load_goes_xray(event, results_dir, time_window)
    sem_df = load_soho_sem(event, results_dir, time_window)
    solar_image = load_solar_image(event, results_dir)

    if no_plots:
        logger.info(f"[{event_name}] Dashboard data available for {nearest_map_time}")
        return {"dashboard": True}

    fig = plt.figure(figsize=PLOT_FIGSIZE_DASHBOARD, constrained_layout=False)
    gs = fig.add_gridspec(
        7, 2,
        height_ratios=[6, 6, 2.2, 2.2, 2.2, 2.2, 2.8],
        width_ratios=[1, 1],
        wspace=0.2, hspace=0.45,
    )

    map_axes = []
    for idx, product in enumerate(PRODUCTS):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        map_axes.append(ax)

        points = map_data.get(product, {}).get(nearest_map_time)
        if points is not None and points.size > 0 and points.dtype.names is not None:
            vmin, vmax = PRODUCT_VMIN_VMAX.get(product, (None, None))
            plot_global_map(ax, points, product, nearest_map_time, vmin=vmin, vmax=vmax)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=LABEL_FONT_SIZE)

    index_axes = []
    for row in range(2, 6):
        ax = fig.add_subplot(gs[row, :])
        index_axes.append(ax)

        idx = row - 2
        product = PRODUCTS[idx]
        df = indices_data.get(product)

        if df is not None and not df.empty:
            ax2 = ax.twinx()
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("axes", 1.1))
            ax3.spines["right"].set_visible(True)

            axes = [ax, ax2, ax3]
            lines = []
            labels = []

            for axis, idx_col in zip(axes, INDEX_COLUMNS):
                if idx_col not in df.columns:
                    continue
                times = df["time"]
                values = pd.to_numeric(df[idx_col], errors="coerce")
                valid = ~values.isna()
                if not valid.any():
                    continue
                color = INDEX_COLORS.get(idx_col, "black")
                line, = axis.plot(times[valid], values[valid], label=INDEX_LABELS.get(idx_col, idx_col),
                                color=color, linewidth=LINE_WIDTH)
                axis.set_ylabel(INDEX_LABELS.get(idx_col, idx_col), color=color, fontsize=LABEL_FONT_SIZE)
                axis.tick_params(axis="y", colors=color, labelsize=TICK_FONT_SIZE)
                axis.spines["left" if axis is ax else "right"].set_color(color)

                if idx_col == "day_night":
                    fill_negative(axis, times[valid], values[valid])

                lines.append(line)
                labels.append(INDEX_LABELS.get(idx_col, idx_col))

            if lines:
                ax.legend(lines, labels, loc="upper left", fontsize=LEGEND_FONT_SIZE, ncol=3)

            add_flare_markers(ax, start_time, peak_time, end_time)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=LABEL_FONT_SIZE)

        apply_grid(ax)
        format_time_axis(ax)
        ax.tick_params(labelsize=TICK_FONT_SIZE)
        if row < 5:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (UTC)", fontsize=LABEL_FONT_SIZE)

    solar_ax = fig.add_subplot(gs[6, 0])
    plot_solar_disk_base(
        solar_ax,
        flare_hpc_x=flare_row.get("hpc_x") if pd.notna(flare_row.get("hpc_x")) else None,
        flare_hpc_y=flare_row.get("hpc_y") if pd.notna(flare_row.get("hpc_y")) else None,
        solar_image=solar_image,
    )

    xray_ax = fig.add_subplot(gs[6, 1])
    xray_ax2 = xray_ax.twinx()

    if not goes_df.empty:
        for col in GOES_XRAY_COLUMNS:
            if col in goes_df.columns:
                color = XRAY_COLORS.get(col, "black")
                xray_ax.semilogy(goes_df["time"], goes_df[col], label=f"GOES {col.upper()}",
                                color=color, linewidth=LINE_WIDTH)
        xray_ax.set_ylabel("Flux (W m$^{-2}$)", fontsize=LABEL_FONT_SIZE)
        xray_ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

    if not sem_df.empty:
        label_map = {"flux_26_34": "SEM 26-34 nm", "flux_01_50": "SEM 0.1-50 nm"}
        for col in SOHO_SEM_COLUMNS:
            if col in sem_df.columns:
                xray_ax2.plot(sem_df["time"], sem_df[col], label=label_map.get(col, col),
                            color=EUV_COLOR, linewidth=LINE_WIDTH, linestyle="--")
        xray_ax2.set_ylabel("EUV Flux (phot. cm$^{-2}$ s$^{-1}$)", fontsize=LABEL_FONT_SIZE)
        xray_ax2.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

    lines1, labels1 = xray_ax.get_legend_handles_labels()
    lines2, labels2 = xray_ax2.get_legend_handles_labels()
    if lines1 or lines2:
        xray_ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=LEGEND_FONT_SIZE)

    add_flare_markers(xray_ax, start_time, peak_time, end_time)
    apply_grid(xray_ax)
    format_time_axis(xray_ax)
    xray_ax.set_xlabel("Time (UTC)", fontsize=LABEL_FONT_SIZE)
    xray_ax.tick_params(labelsize=TICK_FONT_SIZE)

    fig.tight_layout()

    filename = f"dashboard_{nearest_map_time:%H-%M-%S_UTC}.png"
    save_figure(fig, event_name, OUTPUT_SUBDIRS["dashboard"], filename, output_dir)
    logger.info(f"[{event_name}] Saved dashboard for map time {nearest_map_time}")

    return {"dashboard": True}


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

        results = plot_dashboard_for_event(
            event, results_dir, catalog, output_dir,
            args.window_minutes, args.no_plots
        )
        total_processed += 1
        if results.get("dashboard", False):
            total_success += 1
        else:
            logger.warning(f"[{event_name}] Failed to create dashboard")

    logger.info(f"Done. Processed {total_processed} events, {total_success} successful dashboards")


if __name__ == "__main__":
    main()
