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
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import h5py

from .config import (
    PRODUCTS, PRODUCT_LABELS, PRODUCT_VMIN_VMAX, PRODUCT_CMAPS,
    FLARE_CLASSES, PLOT_FIGSIZE_DASHBOARD, PLOT_DPI, MAP_POINT_SIZE, MAP_ALPHA,
    SOLAR_RADIUS_ARCSEC, TIME_WINDOW_MINUTES,
    DEFAULT_RESULTS_DIR, DEFAULT_FLARES_CSV, DEFAULT_OUTPUT_DIR,
    OUTPUT_SUBDIRS, GOES_XRAY_COLUMNS, SOHO_SEM_COLUMNS, INDEX_COLUMNS,
)
from .utils import (
    load_events, load_flare_catalog, event_file_path,
    normalize_time_column, get_flare_time_window, find_flare_row,
    load_solar_image, plot_solar_disk_base, plot_global_map,
    add_flare_markers, format_time_axis, subsolar_point, plot_terminator,
    save_figure, logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def load_hdf5_map_data(event: dict, results_dir: Path, product: str, time_window: tuple) -> tuple:
    path = event_file_path(results_dir, event, "maps", f"map_{product}.h5")
    if not path.exists():
        return [], {}

    timestamps = []
    product_data = {}

    try:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                return [], {}
            data_group = f["data"]
            for str_time in data_group.keys():
                try:
                    time = pd.Timestamp(str_time, tz='UTC')
                except Exception:
                    continue
                if time_window[0] <= time <= time_window[1]:
                    try:
                        values = data_group[str_time][:]
                        if product == "roti":
                            mask = (values['vals'] >= 0) & (values['vals'] <= 5)
                            values = values[mask]
                    except Exception:
                        continue
                    timestamps.append(time)
                    product_data[time] = values
    except Exception as exc:
        logger.warning(f"Failed to read HDF5 map {path}: {exc}")
        return [], {}

    return sorted(timestamps), product_data


def load_indices_data(event: dict, results_dir: Path, product: str, time_window: tuple) -> pd.DataFrame:
    path = event_file_path(results_dir, event, "indices", f"indices_{product}.csv")
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = normalize_time_column(df, preferred="time")
    mask = (df["time"] >= time_window[0]) & (df["time"] <= time_window[1])
    return df[mask].reset_index(drop=True)


def load_goes_data(event: dict, results_dir: Path, time_window: tuple) -> pd.DataFrame:
    path = event_file_path(results_dir, event, "goes_xray", "goes_xray.csv")
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = normalize_time_column(df, preferred="time")
    for col in GOES_XRAY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = (df["time"] >= time_window[0]) & (df["time"] <= time_window[1])
    return df[mask].reset_index(drop=True)


def load_sem_data(event: dict, results_dir: Path, time_window: tuple) -> pd.DataFrame:
    path = event_file_path(results_dir, event, "soho_sem", "soho_sem.csv")
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = normalize_time_column(df, preferred="time")
    for col in SOHO_SEM_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = (df["time"] >= time_window[0]) & (df["time"] <= time_window[1])
    return df[mask].reset_index(drop=True)


def find_nearest_map_time(timestamps: list, target_time: pd.Timestamp, tolerance_minutes: float = 5.0):
    if not timestamps:
        return None
    target = target_time.tz_convert(None) if target_time.tz else target_time
    nearest = min(timestamps, key=lambda t: abs((t.tz_convert(None) if t.tz else t) - target).total_seconds())
    delta = abs((nearest.tz_convert(None) if nearest.tz else nearest) - target).total_seconds() / 60.0
    if delta <= tolerance_minutes:
        return nearest
    return None


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

    # Load data for all products
    map_timestamps = {}
    map_data = {}
    for product in PRODUCTS:
        if event.get("maps", {}).get(product):
            timestamps, data = load_hdf5_map_data(event, results_dir, product, time_window)
            if timestamps:
                map_timestamps[product] = timestamps
                map_data[product] = data

    # Find common timestamps across all products
    all_timestamps = set()
    for ts in map_timestamps.values():
        all_timestamps.update(ts)
    all_timestamps = sorted(all_timestamps)

    if not all_timestamps:
        logger.warning(f"[{event_name}] No map timestamps in window")
        return {"dashboard": False}

    # Find nearest map time to flare peak
    nearest_map_time = find_nearest_map_time(all_timestamps, peak_time, tolerance_minutes=window_minutes)
    if nearest_map_time is None:
        nearest_map_time = all_timestamps[0]
        logger.warning(f"[{event_name}] No map time near peak, using first: {nearest_map_time}")

    # Load indices for the nearest map time
    index_time_window = get_flare_time_window(nearest_map_time, window_minutes=5.0)
    indices_data = {}
    for product in PRODUCTS:
        df = load_indices_data(event, results_dir, product, index_time_window)
        if not df.empty:
            indices_data[product] = df

    # Load GOES and SOHO data
    goes_df = load_goes_data(event, results_dir, time_window)
    sem_df = load_sem_data(event, results_dir, time_window)

    solar_image = load_solar_image(event, results_dir)

    if no_plots:
        logger.info(f"[{event_name}] Dashboard data available for {nearest_map_time}")
        return {"dashboard": True}

    # Create dashboard figure
    fig = plt.figure(figsize=PLOT_FIGSIZE_DASHBOARD, constrained_layout=False)
    gs = fig.add_gridspec(
        7, 2,
        height_ratios=[6, 6, 2.2, 2.2, 2.2, 2.2, 2.8],
        width_ratios=[1, 1],
        wspace=0.2, hspace=0.45,
    )

    # Top row: 4 maps
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
            ax.set_title(f"No data for {PRODUCT_LABELS[product]}")

    # Index panels (rows 2-5)
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
            index_cols = ["day_night_index", "gsflai_index", "isfai_index"]
            colors = ["tab:blue", "tab:green", "tab:red"]

            lines = []
            labels = []

            for axis, idx_col, color in zip(axes, index_cols, colors):
                if idx_col not in df.columns:
                    continue
                times = df["time"]
                values = pd.to_numeric(df[idx_col], errors="coerce")
                valid = ~values.isna()
                if not valid.any():
                    continue
                line, = axis.plot(times[valid], values[valid], label=idx_col, color=color, linewidth=1.3)
                axis.set_ylabel(idx_col, color=color, fontsize=9)
                axis.tick_params(axis="y", colors=color, labelsize=8)
                axis.spines["left" if axis is ax else "right"].set_color(color)
                lines.append(line)
                labels.append(idx_col)

            if lines:
                ax.legend(lines, labels, loc="upper left", fontsize=8, ncol=3)

            add_flare_markers(ax, start_time, peak_time, end_time)
        else:
            ax.set_title(f"No indices for {PRODUCT_LABELS[product]}")

        format_time_axis(ax)
        if row < 5:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (UTC)")

    # Solar disk + X-ray/EUV (bottom row)
    solar_ax = fig.add_subplot(gs[6, 0])
    plot_solar_disk_base(
        solar_ax,
        flare_hpc_x=flare_row.get("hpc_x") if pd.notna(flare_row.get("hpc_x")) else None,
        flare_hpc_y=flare_row.get("hpc_y") if pd.notna(flare_row.get("hpc_y")) else None,
        solar_image=solar_image,
        title="Solar Disk",
    )

    xray_ax = fig.add_subplot(gs[6, 1])
    xray_ax2 = xray_ax.twinx()

    if not goes_df.empty:
        for col in GOES_XRAY_COLUMNS:
            if col in goes_df.columns:
                xray_ax.semilogy(goes_df["time"], goes_df[col], label=f"GOES {col.upper()}", linewidth=1.2)
        xray_ax.set_ylabel("X-ray Flux (W/m\u00b2)", color="purple", fontsize=9)
        xray_ax.tick_params(axis="y", colors="purple", labelsize=8)
        xray_ax.spines["left"].set_color("purple")

    if not sem_df.empty:
        for col in SOHO_SEM_COLUMNS:
            if col in sem_df.columns:
                xray_ax2.semilogy(sem_df["time"], sem_df[col], label=f"SEM {col}", linewidth=1.2, linestyle="--")
        xray_ax2.set_ylabel("EUV Flux", color="brown", fontsize=9)
        xray_ax2.tick_params(axis="y", colors="brown", labelsize=8)
        xray_ax2.spines["right"].set_color("brown")

    lines1, labels1 = xray_ax.get_legend_handles_labels()
    lines2, labels2 = xray_ax2.get_legend_handles_labels()
    if lines1 or lines2:
        xray_ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    add_flare_markers(xray_ax, start_time, peak_time, end_time)
    format_time_axis(xray_ax)
    xray_ax.set_xlabel("Time (UTC)")

    # Title
    fig.suptitle(
        f"{event_name} | {flare_class}-class Flare | Peak: {peak_time:%Y-%m-%d %H:%M UTC} | Map: {nearest_map_time:%H:%M UTC}",
        fontsize=14, fontweight="bold", y=0.98
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.93, right=0.92)

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