#!/usr/bin/env python3
"""
Combined dashboard: ROTI map + solar disk + GOES X-ray + SOHO SEM EUV.
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

from .config import (
    PRODUCTS, PRODUCT_LABELS, PRODUCT_VMIN_VMAX, PRODUCT_CMAPS,
    FLARE_CLASSES, PLOT_FIGSIZE_DASHBOARD, PLOT_DPI, MAP_POINT_SIZE, MAP_ALPHA,
    SOLAR_RADIUS_ARCSEC, TIME_WINDOW_MINUTES,
    DEFAULT_RESULTS_DIR, DEFAULT_FLARES_CSV, DEFAULT_OUTPUT_DIR,
    OUTPUT_SUBDIRS, GOES_XRAY_COLUMNS, SOHO_SEM_COLUMNS,
    LINE_WIDTH, XRAY_COLORS, EUV_COLOR,
    LABEL_FONT_SIZE, TICK_FONT_SIZE, LEGEND_FONT_SIZE, TITLE_FONT_SIZE,
)
from .utils import (
    load_events, load_flare_catalog, event_file_path,
    normalize_time_column, get_flare_time_window, find_flare_row,
    load_solar_image, plot_solar_disk_base, plot_global_map,
    add_flare_markers, format_time_axis, apply_grid,
    load_hdf5_map, load_goes_xray, load_soho_sem,
    find_nearest_map_time, save_figure, logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PANEL_LABELS_CHARS = ["A", "B", "C"]


def smooth_series(t: pd.Series, y: pd.Series, window: int = 5) -> tuple:
    """Simple moving average smoothing."""
    if len(y) < window:
        return t, y
    s = y.rolling(window=window, center=True, min_periods=1).mean()
    return t, s


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
    start_time = flare_row.get("start_time")
    end_time = flare_row.get("end_time")

    timestamps, map_data = load_hdf5_map(event, results_dir, "roti", time_window)
    if not timestamps:
        logger.warning(f"[{event_name}] No ROTI map timestamps in window")
        return {"dashboard": False}

    nearest_map_time = find_nearest_map_time(timestamps, peak_time, tolerance_minutes=window_minutes)
    if nearest_map_time is None:
        nearest_map_time = timestamps[0]
        logger.warning(f"[{event_name}] No ROTI map near peak, using first: {nearest_map_time}")

    goes_df = load_goes_xray(event, results_dir, time_window)
    sem_df = load_soho_sem(event, results_dir, time_window)
    solar_image = load_solar_image(event, results_dir)

    if no_plots:
        logger.info(f"[{event_name}] Dashboard data available for {nearest_map_time}")
        return {"dashboard": True}

    fig = plt.figure(figsize=PLOT_FIGSIZE_DASHBOARD, constrained_layout=False)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[4.5, 4.5],
        width_ratios=[1, 1],
        wspace=0.3, hspace=0.35,
    )
    gs.update(left=0.06, right=0.95, top=0.85, bottom=0.18)

    fig.suptitle(
        f"{event_name} ({flare_row['class']}-class flare)\n"
        f"Peak: {peak_time:%Y-%m-%d %H:%M:%S UTC}",
        fontsize=18,
        fontweight="bold",
        y=1.02,
    )

    panels = []

    # --- Panel A: ROTI map ---
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    points = map_data.get(nearest_map_time)
    if points is not None and points.size > 0 and points.dtype.names is not None:
        cbar = plot_global_map(ax_map, points, "roti", nearest_map_time, vmin=0.0, vmax=0.3)
        if cbar:
            cbar.ax.tick_params(labelsize=11)
            cbar.set_label("ROTI (TECU min$^{-1}$)", fontsize=12)
    else:
        ax_map.text(0.5, 0.5, "No data", transform=ax_map.transAxes,
                   ha="center", va="center", fontsize=LABEL_FONT_SIZE)
    ax_map.set_title("Global ROTI map", fontsize=15, fontweight="bold")
    panels.append(ax_map)

    # --- Panel B: Solar disk ---
    ax_solar = fig.add_subplot(gs[0, 1])
    hpc_x = flare_row.get("hpc_x") if pd.notna(flare_row.get("hpc_x")) else None
    hpc_y = flare_row.get("hpc_y") if pd.notna(flare_row.get("hpc_y")) else None
    plot_solar_disk_base(ax_solar, flare_hpc_x=hpc_x, flare_hpc_y=hpc_y, solar_image=solar_image)
    ax_solar.set_title("Solar disk", fontsize=15, fontweight="bold")
    if hpc_x is not None and hpc_y is not None:
        ax_solar.text(0.5, -0.05,
                     f"HPC: ({hpc_x:.0f}, {hpc_y:.0f})\"",
                     transform=ax_solar.transAxes, ha="center", fontsize=12)
    panels.append(ax_solar)

    # --- Panel C: GOES X-ray + SOHO SEM EUV flux ---
    ax_xray = fig.add_subplot(gs[1, :])
    ax_xray2 = ax_xray.twinx()

    if not goes_df.empty:
        labels_map = {"xrsa": "GOES XRS-A (0.05\u20130.4 nm)", "xrsb": "GOES XRS-B (0.1\u20130.8 nm)"}
        for col in GOES_XRAY_COLUMNS:
            if col in goes_df.columns:
                color = XRAY_COLORS.get(col, "black")
                t, y = smooth_series(goes_df["time"], goes_df[col], window=5)
                ax_xray.semilogy(t, y, label=labels_map.get(col, col.upper()),
                                color=color, linewidth=LINE_WIDTH)
        ax_xray.set_ylabel("Flux (W m$^{-2}$)", fontsize=LABEL_FONT_SIZE, color="black", labelpad=15)
        ax_xray.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

    if not sem_df.empty:
        labels_map = {"flux_26_34": "SOHO/SEM 26\u201334 nm", "flux_01_50": "SOHO/SEM 0.1\u201350 nm"}
        for col in SOHO_SEM_COLUMNS:
            if col in sem_df.columns:
                t, y = smooth_series(sem_df["time"], sem_df[col], window=3)
                ax_xray2.plot(t, y, label=labels_map.get(col, col),
                            color=EUV_COLOR, linewidth=LINE_WIDTH, linestyle="--")
        ax_xray2.set_ylabel("EUV (phot. cm$^{-2}$ s$^{-1}$)", fontsize=LABEL_FONT_SIZE, labelpad=15)
        ax_xray2.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

    # Legend outside
    lines1, labels1 = ax_xray.get_legend_handles_labels()
    lines2, labels2 = ax_xray2.get_legend_handles_labels()
    if lines1 or lines2:
        ax_xray.legend(lines1 + lines2, labels1 + labels2,
                      loc="upper center", bbox_to_anchor=(0.5, -0.15),
                      fontsize=LEGEND_FONT_SIZE, framealpha=0.8,
                      ncol=2, borderaxespad=0.)

    # Flare markers on X-ray panel
    add_flare_markers(ax_xray, start_time, peak_time, end_time, peak_lw=1.5)

    # X-axis: limit, ticks, grid
    ax_xray.grid(True, which="both", alpha=0.25)
    ax_xray.set_xlim(time_window[0], time_window[1])
    ax_xray.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax_xray.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    ax_xray.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_xray.set_xlabel("Time (UTC)", fontsize=LABEL_FONT_SIZE)
    ax_xray.set_title("GOES X-ray and SOHO/SEM EUV flux", fontsize=14)
    ax_xray.tick_params(labelsize=TICK_FONT_SIZE)

    panels.append(ax_xray)

    # --- Panel labels A, B, C ---
    for ax, label in zip(panels, PANEL_LABELS_CHARS):
        ax.text(0.01, 0.99, label, transform=ax.transAxes,
               fontsize=18, fontweight="bold", va="top", ha="left",
               bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    fig.tight_layout(rect=[0, 0, 1, 0.95])

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

    if not results_dir.exists() or not flares_csv.exists():
        logger.error("Results dir or flares CSV not found")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(results_dir)
    catalog = load_flare_catalog(flares_csv)

    events_to_process = events[:args.max_events] if args.max_events else events
    total = success = 0
    for event in events_to_process:
        if event.get("class") not in args.flare_classes:
            continue
        results = plot_dashboard_for_event(
            event, results_dir, catalog, output_dir,
            args.window_minutes, args.no_plots
        )
        total += 1
        if results.get("dashboard", False):
            success += 1
        else:
            logger.warning(f"[{event.get('name')}] Failed to create dashboard")

    logger.info(f"Done. {success}/{total} successful dashboards")


if __name__ == "__main__":
    main()
