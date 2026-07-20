#!/usr/bin/env python3
"""
Aggregate statistics and plots across all flares.
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
    PRODUCTS, PRODUCT_LABELS, FLARE_CLASSES, FLARE_CLASS_MARKERS, FLARE_CLASS_COLORS,
    PLOT_FIGSIZE_STATS, PLOT_DPI, DEFAULT_RESULTS_DIR, DEFAULT_FLARES_CSV,
    DEFAULT_OUTPUT_DIR, OUTPUT_SUBDIRS, TIME_WINDOW_MINUTES,
)
from .utils import (
    load_events, load_flare_catalog, event_file_path,
    get_flare_time_window, find_flare_row, load_hdf5_map,
    logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute aggregate statistics across all flares")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--flares-csv", type=Path, default=DEFAULT_FLARES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flare-classes", nargs="+", default=list(FLARE_CLASSES), choices=FLARE_CLASSES)
    parser.add_argument("--window-minutes", type=float, default=TIME_WINDOW_MINUTES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def compute_flare_metrics(event: dict, results_dir: Path, catalog: pd.DataFrame, window_minutes: float) -> dict | None:
    flare_row = find_flare_row(event, catalog)
    if flare_row is None:
        return None

    peak_time = flare_row.get("peak_time")
    if pd.isna(peak_time):
        return None

    time_window = get_flare_time_window(peak_time, window_minutes)
    event_name = event.get("name", "unknown")
    flare_class = str(flare_row.get("class", event.get("class", "?"))).upper()[0]

    metrics = {
        "event": event_name,
        "flare_class": flare_class,
        "peak_time": peak_time,
        "start_time": flare_row.get("start_time"),
        "end_time": flare_row.get("end_time"),
        "duration_min": flare_row.get("duration_min"),
        "hpc_x": flare_row.get("hpc_x"),
        "hpc_y": flare_row.get("hpc_y"),
        "peak_flux": flare_row.get("peak_flux"),
    }

    for product in PRODUCTS:
        if not event.get("maps", {}).get(product):
            metrics[f"{product}_has_map"] = False
            continue

        timestamps, product_data = load_hdf5_map(event, results_dir, product, time_window)
        if not timestamps:
            metrics[f"{product}_has_map"] = False
            continue

        nearest_time = min(timestamps, key=lambda t: abs((t - peak_time).total_seconds()))
        points = product_data.get(nearest_time)

        metrics[f"{product}_has_map"] = True
        metrics[f"{product}_map_time"] = nearest_time
        metrics[f"{product}_time_diff_min"] = abs((nearest_time - peak_time).total_seconds()) / 60.0

        if points is not None and points.size > 0 and points.dtype.names is not None:
            vals = points['vals']
            metrics[f"{product}_n_points"] = len(vals)
            metrics[f"{product}_mean"] = float(np.nanmean(vals))
            metrics[f"{product}_std"] = float(np.nanstd(vals))
            metrics[f"{product}_max"] = float(np.nanmax(vals))
            metrics[f"{product}_median"] = float(np.nanmedian(vals))
            metrics[f"{product}_q95"] = float(np.nanpercentile(vals, 95))

    if event.get("sources", {}).get("goes_xray"):
        path = event_file_path(results_dir, event, "goes_xray", "goes_xray.csv")
        if path.exists():
            df = pd.read_csv(path)
            df = df.rename(columns={df.columns[0]: "time"})
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            for col in ["xrsa", "xrsb"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            mask = (df["time"] >= time_window[0]) & (df["time"] <= time_window[1])
            window_df = df[mask]
            if not window_df.empty:
                for col in ["xrsa", "xrsb"]:
                    if col in window_df.columns:
                        metrics[f"goes_{col}_max"] = float(np.nanmax(window_df[col]))
                        metrics[f"goes_{col}_mean"] = float(np.nanmean(window_df[col]))

    if event.get("sources", {}).get("soho_sem"):
        path = event_file_path(results_dir, event, "soho_sem", "soho_sem.csv")
        if path.exists():
            df = pd.read_csv(path)
            df = df.rename(columns={df.columns[0]: "time"})
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            for col in ["flux_26_34", "flux_01_50"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            mask = (df["time"] >= time_window[0]) & (df["time"] <= time_window[1])
            window_df = df[mask]
            if not window_df.empty:
                for col in ["flux_26_34", "flux_01_50"]:
                    if col in window_df.columns:
                        metrics[f"sem_{col}_max"] = float(np.nanmax(window_df[col]))
                        metrics[f"sem_{col}_mean"] = float(np.nanmean(window_df[col]))

    return metrics


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
    stats_dir = output_dir / OUTPUT_SUBDIRS["statistics"]
    stats_dir.mkdir(parents=True, exist_ok=True)

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

    all_metrics = []

    for event in events_to_process:
        flare_class = event.get("class", "?")
        if flare_class not in args.flare_classes:
            continue

        metrics = compute_flare_metrics(event, results_dir, catalog, args.window_minutes)
        if metrics:
            all_metrics.append(metrics)

    if not all_metrics:
        logger.warning("No metrics computed")
        return

    df = pd.DataFrame(all_metrics)
    csv_path = stats_dir / "flare_statistics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved statistics to {csv_path}")

    if args.no_plots:
        logger.info("Skipping plots")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    # Plot 1: Peak ROTI vs Flare Class
    if "roti_max" in df.columns:
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_STATS)
        for fc in FLARE_CLASSES:
            subset = df[df["flare_class"] == fc]
            if subset.empty:
                continue
            ax.scatter(
                subset["peak_flux"], subset["roti_max"],
                s=80, alpha=0.8, marker=FLARE_CLASS_MARKERS[fc],
                color=FLARE_CLASS_COLORS[fc], label=f"{fc}-class"
            )
        ax.set_xscale("log")
        ax.set_xlabel("GOES Peak Flux (W/m\u00b2)")
        ax.set_ylabel("Peak ROTI (TECu/min)")
        ax.set_title("Peak ROTI vs Flare Class")
        ax.legend(title="Flare Class")
        fig.tight_layout()
        fig.savefig(stats_dir / "peak_roti_vs_flare_class.png", dpi=PLOT_DPI)
        plt.close(fig)

    # Plot 2: Flare position distribution on solar disk
    pos_df = df.dropna(subset=["hpc_x", "hpc_y"])
    if not pos_df.empty:
        fig, ax = plt.subplots(figsize=(8, 8))
        disk = plt.Circle((0, 0), SOLAR_RADIUS_ARCSEC, color="black", fill=False, linewidth=1.5, alpha=0.7)
        ax.add_patch(disk)
        for fc in FLARE_CLASSES:
            subset = pos_df[pos_df["flare_class"] == fc]
            if subset.empty:
                continue
            ax.scatter(
                subset["hpc_x"], subset["hpc_y"],
                s=80, alpha=0.8, marker=FLARE_CLASS_MARKERS[fc],
                color=FLARE_CLASS_COLORS[fc], label=f"{fc}-class"
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("HPC X (arcsec)")
        ax.set_ylabel("HPC Y (arcsec)")
        ax.set_title("Flare Positions on Solar Disk")
        ax.legend(title="Flare Class")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(stats_dir / "flare_positions_solar_disk.png", dpi=PLOT_DPI)
        plt.close(fig)

    # Plot 3: ROTI distribution by flare class
    if "roti_max" in df.columns:
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_STATS)
        data = [df[df["flare_class"] == fc]["roti_max"].dropna().values for fc in FLARE_CLASSES if not df[df["flare_class"] == fc]["roti_max"].dropna().empty]
        labels = [fc for fc in FLARE_CLASSES if not df[df["flare_class"] == fc]["roti_max"].dropna().empty]
        if data:
            ax.boxplot(data, labels=labels, showfliers=True)
            ax.set_ylabel("Peak ROTI (TECu/min)")
            ax.set_title("ROTI Distribution by Flare Class")
            fig.tight_layout()
            fig.savefig(stats_dir / "roti_distribution_by_class.png", dpi=PLOT_DPI)
            plt.close(fig)

    # Plot 4: Number of map points per product by flare class
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for idx, product in enumerate(PRODUCTS):
        ax = axes[idx]
        data = []
        labels = []
        for fc in FLARE_CLASSES:
            col = f"{product}_n_points"
            if col in df.columns:
                vals = df[(df["flare_class"] == fc) & df[col].notna()][col]
                if not vals.empty:
                    data.append(vals)
                    labels.append(fc)
        if data:
            ax.boxplot(data, labels=labels, showfliers=True)
        ax.set_title(f"{PRODUCT_LABELS[product]}: N Points")
        ax.set_ylabel("Number of Points")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Map Point Count by Product and Flare Class", fontsize=14)
    fig.tight_layout()
    fig.savefig(stats_dir / "n_points_by_product_class.png", dpi=PLOT_DPI)
    plt.close(fig)

    # Plot 5: Time difference between map and flare peak
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE_STATS)
    for product in PRODUCTS:
        col = f"{product}_time_diff_min"
        if col in df.columns:
            vals = df[col].dropna()
            if not vals.empty:
                ax.hist(vals, bins=20, alpha=0.5, label=PRODUCT_LABELS[product], density=True)
    ax.set_xlabel("Time Difference (minutes)")
    ax.set_ylabel("Density")
    ax.set_title("Map Timestamp vs Flare Peak Time Difference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(stats_dir / "map_time_diff_distribution.png", dpi=PLOT_DPI)
    plt.close(fig)

    # Plot 6: Summary table as image
    summary = df.groupby("flare_class").agg(
        n_events=("event", "count"),
        mean_duration=("duration_min", "mean"),
        mean_roti_max=("roti_max", "mean") if "roti_max" in df.columns else ("event", "count"),
        mean_hpc_x=("hpc_x", "mean"),
        mean_hpc_y=("hpc_y", "mean"),
    ).round(2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        rowLabels=summary.index,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    fig.suptitle("Summary Statistics by Flare Class", fontsize=14)
    fig.tight_layout()
    fig.savefig(stats_dir / "summary_table.png", dpi=PLOT_DPI)
    plt.close(fig)

    logger.info(f"Saved plots to {stats_dir}")


if __name__ == "__main__":
    main()