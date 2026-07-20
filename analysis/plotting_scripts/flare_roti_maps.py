#!/usr/bin/env python3
"""
Plot global ROTI/dTEC maps at flare peak time for each event.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from config import (
    PRODUCTS, PRODUCT_LABELS, PRODUCT_VMIN_VMAX, PRODUCT_CMAPS,
    FLARE_CLASSES, TIME_WINDOW_MINUTES, PLOT_FIGSIZE_SINGLE,
    PLOT_DPI, DEFAULT_RESULTS_DIR, DEFAULT_FLARES_CSV, DEFAULT_OUTPUT_DIR,
    OUTPUT_SUBDIRS,
)
from utils import (
    load_events, find_event_by_name, event_file_path, load_hdf5_map,
    get_flare_time_window, find_nearest_map_time, plot_global_map,
    load_flare_catalog, load_solar_image, save_figure, logger,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot global ionospheric maps at flare peak time")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--flares-csv", type=Path, default=DEFAULT_FLARES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--products", nargs="+", default=list(PRODUCTS), choices=PRODUCTS)
    parser.add_argument("--window-minutes", type=float, default=TIME_WINDOW_MINUTES)
    parser.add_argument("--flare-classes", nargs="+", default=list(FLARE_CLASSES), choices=FLARE_CLASSES)
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting, only check data availability")
    parser.add_argument("--max-events", type=int, default=None, help="Limit number of events to process")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def find_flare_peak_time(event: dict, catalog: pd.DataFrame) -> pd.Timestamp | None:
    if "flare_key" in catalog.columns:
        match = catalog[catalog["flare_key"].astype(str) == event.get("name", "")]
        if len(match) == 1 and pd.notna(match.iloc[0].get("peak_time")):
            return match.iloc[0]["peak_time"]

    event_name = event.get("name", "")
    if "_" in event_name:
        parts = event_name.split("_")
        if len(parts) >= 2:
            date_str = parts[0]
            class_str = parts[1]
            match = catalog[(catalog["date"] == date_str) & (catalog["class"] == class_str)]
            if len(match) == 1 and pd.notna(match.iloc[0].get("peak_time")):
                return match.iloc[0]["peak_time"]

    return None


def plot_maps_for_event(
    event: dict,
    results_dir: Path,
    catalog: pd.DataFrame,
    products: list[str],
    output_dir: Path,
    window_minutes: float,
    no_plots: bool,
) -> dict[str, bool]:
    flare_peak = find_flare_peak_time(event, catalog)
    if flare_peak is None:
        logger.warning(f"[{event.get('name')}] No flare peak time found in catalog")
        return {p: False for p in products}

    time_window = get_flare_time_window(flare_peak, window_minutes)
    event_name = event.get("name", "unknown")
    logger.info(f"[{event_name}] Flare peak: {flare_peak}, window: {time_window[0]} - {time_window[1]}")

    results = {}
    for product in products:
        if not event.get("maps", {}).get(product):
            logger.warning(f"[{event_name}] No map data for product {product}")
            results[product] = False
            continue

        timestamps, product_data = load_hdf5_map(event, results_dir, product, time_window)
        if not timestamps:
            logger.warning(f"[{event_name}] No map timestamps in window for {product}")
            results[product] = False
            continue

        nearest_time = find_nearest_map_time(timestamps, flare_peak, tolerance_minutes=window_minutes)
        if nearest_time is None:
            logger.warning(f"[{event_name}] No map time near flare peak for {product}")
            results[product] = False
            continue

        points = product_data.get(nearest_time)
        if points is None or points.size == 0:
            logger.warning(f"[{event_name}] Empty map data at {nearest_time} for {product}")
            results[product] = False
            continue

        if no_plots:
            logger.info(f"[{event_name}] {product}: map at {nearest_time} has {len(points)} points")
            results[product] = True
            continue

        vmin, vmax = PRODUCT_VMIN_VMAX.get(product, (None, None))
        fig = plt.figure(figsize=PLOT_FIGSIZE_SINGLE)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        plot_global_map(ax, points, product, nearest_time, vmin=vmin, vmax=vmax)

        flare_class = event.get("class", "?")
        fig.suptitle(
            f"{event_name}  |  Flare: {flare_class}-class  |  Peak: {flare_peak:%H:%M UTC}  |  Map: {nearest_time:%H:%M UTC}",
            fontsize=14, fontweight="bold", y=0.95
        )
        fig.tight_layout()

        filename = f"map_{product}_{nearest_time:%H-%M-%S_UTC}.png"
        save_figure(fig, event_name, OUTPUT_SUBDIRS["maps"], filename, output_dir)
        results[product] = True
        logger.info(f"[{event_name}] Saved {product} map at {nearest_time}")

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

        results = plot_maps_for_event(
            event, results_dir, catalog, args.products,
            output_dir, args.window_minutes, args.no_plots
        )

        success_count = sum(1 for v in results.values() if v)
        total_success += success_count
        total_processed += 1

        if success_count == 0:
            logger.warning(f"[{event_name}] No products plotted")
        else:
            logger.info(f"[{event_name}] Success: {success_count}/{len(args.products)} products")

    logger.info(f"Done. Processed {total_processed} events, {total_success} successful product plots")


if __name__ == "__main__":
    main()