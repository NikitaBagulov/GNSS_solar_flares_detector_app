from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

PRODUCTS = ("roti", "dtec_2_10", "dtec_10_20", "dtec_20_60")
INDEX_COLUMNS = ("day_night_index", "gsflai_index", "isfai_index")


def load_events(results_dir: Path) -> list[dict]:
    sys.path.insert(0, str(REPO_ROOT))
    from results_server import scan_events

    return scan_events(results_dir)


def normalize_time_column(df: pd.DataFrame, preferred: str = "time") -> pd.DataFrame:
    df = df.copy()
    if preferred not in df.columns:
        df = df.rename(columns={df.columns[0]: preferred})
    df[preferred] = pd.to_datetime(df[preferred], utc=True, errors="coerce")
    return df.dropna(subset=[preferred]).sort_values(preferred)


def event_file_path(results_dir: Path, event: dict, *parts: str) -> Path:
    return results_dir / event["path"] / Path(*parts)


def load_goes(results_dir: Path, event: dict, xray_column: str) -> pd.DataFrame:
    path = event_file_path(results_dir, event, "goes_xray", "goes_xray.csv")
    df = normalize_time_column(pd.read_csv(path), preferred="time")
    for column in ("xrsa", "xrsb", xray_column):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def load_indices(results_dir: Path, event: dict, product: str) -> pd.DataFrame:
    path = event_file_path(results_dir, event, "indices", f"indices_{product}.csv")
    df = normalize_time_column(pd.read_csv(path), preferred="time")
    for column in INDEX_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def parse_event_peak_time(event: dict) -> pd.Timestamp | None:
    timestamps = re.findall(r"\d{8}T\d{6}", event.get("name", ""))
    if len(timestamps) < 2:
        return None
    return pd.to_datetime(timestamps[1], format="%Y%m%dT%H%M%S", utc=True, errors="coerce")


def goes_peak(goes: pd.DataFrame, xray_column: str) -> pd.Series:
    if xray_column not in goes.columns:
        raise ValueError(f"GOES CSV has no column {xray_column!r}")
    values = pd.to_numeric(goes[xray_column], errors="coerce")
    if values.dropna().empty:
        raise ValueError(f"GOES column {xray_column!r} has no numeric data")
    return goes.loc[values.idxmax()]


def nearest_row(
    frame: pd.DataFrame,
    target_time: pd.Timestamp,
    tolerance: pd.Timedelta,
) -> pd.Series | None:
    if frame.empty:
        return None
    deltas = (frame["time"] - target_time).abs()
    nearest_idx = deltas.idxmin()
    if deltas.loc[nearest_idx] > tolerance:
        return None
    return frame.loc[nearest_idx]


def build_statistics(
    results_dir: Path,
    events: list[dict],
    xray_column: str,
    peak_time_source: str,
    max_time_delta: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    errors: list[dict] = []
    usable_events = [
        event
        for event in events
        if event.get("sources", {}).get("goes_xray") and any(event.get("indices", {}).values())
    ]

    print(f"Events in results: {len(events)}")
    print(f"Events with GOES and at least one index file: {len(usable_events)}")

    for event_idx, event in enumerate(usable_events, 1):
        if event_idx == 1 or event_idx % 10 == 0 or event_idx == len(usable_events):
            print(f"[{event_idx}/{len(usable_events)}] {event.get('name')}")

        try:
            goes = load_goes(results_dir, event, xray_column)
            flare_peak_time = parse_event_peak_time(event) if peak_time_source == "event_name" else None
            if flare_peak_time is not None and pd.notna(flare_peak_time):
                peak = nearest_row(goes, flare_peak_time, max_time_delta)
                if peak is None:
                    raise ValueError("no GOES measurement near event peak time")
                peak_source = "event_name"
            else:
                peak = goes_peak(goes, xray_column)
                flare_peak_time = peak["time"]
                peak_source = "goes_max"
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            errors.append({"event": event.get("name"), "stage": "goes", "error": str(exc)})
            continue

        for product in PRODUCTS:
            if not event.get("indices", {}).get(product):
                continue
            try:
                indices = load_indices(results_dir, event, product)
                nearest = nearest_row(indices, flare_peak_time, max_time_delta)
                if nearest is None:
                    errors.append(
                        {
                            "event": event.get("name"),
                            "product": product,
                            "stage": "nearest",
                            "error": "no index row near event peak time",
                        }
                    )
                    continue

                row = {
                    "event": event["name"],
                    "event_path": event["path"],
                    "flare_class": event.get("class"),
                    "product": product,
                    "flare_peak_time": flare_peak_time,
                    "goes_time": peak["time"],
                    "peak_source": peak_source,
                    "index_time": nearest["time"],
                    "index_time_delta_seconds": abs((nearest["time"] - flare_peak_time).total_seconds()),
                    "goes_time_delta_seconds": abs((peak["time"] - flare_peak_time).total_seconds()),
                    "xray_column": xray_column,
                    "xray_at_flare_peak": float(peak[xray_column]),
                }
                for column in INDEX_COLUMNS:
                    row[column] = float(nearest[column]) if column in nearest.index and pd.notna(nearest[column]) else np.nan
                rows.append(row)
            except (ValueError, OSError, pd.errors.ParserError) as exc:
                errors.append(
                    {
                        "event": event.get("name"),
                        "product": product,
                        "stage": "indices",
                        "error": str(exc),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(errors)


def build_correlations(stats: pd.DataFrame) -> pd.DataFrame:
    corr_rows = []
    if stats.empty:
        return pd.DataFrame(columns=["product", "index", "n", "spearman_r"])

    for product, product_df in stats.groupby("product"):
        for index_column in INDEX_COLUMNS:
            subset = product_df.dropna(subset=["xray_at_flare_peak", index_column])
            corr = np.nan
            if len(subset) >= 2:
                corr = subset[["xray_at_flare_peak", index_column]].corr(method="spearman").iloc[0, 1]
            corr_rows.append(
                {
                    "product": product,
                    "index": index_column,
                    "n": len(subset),
                    "spearman_r": corr,
                }
            )
    return pd.DataFrame(corr_rows).sort_values(["index", "product"])


def plot_index_vs_xray(stats: pd.DataFrame, index_column: str, xray_column: str, output_dir: Path) -> None:
    data = stats.dropna(subset=["xray_at_flare_peak", index_column]).copy()
    if data.empty:
        print(f"No plot data for {index_column}")
        return

    products = [product for product in PRODUCTS if product in set(data["product"])]
    cols = 2
    rows_count = math.ceil(len(products) / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(13, 4.8 * rows_count), squeeze=False)
    axes_flat = axes.ravel()

    for ax, product in zip(axes_flat, products):
        subset = data[data["product"] == product]
        ax.scatter(subset["xray_at_flare_peak"], subset[index_column], s=52, alpha=0.82)
        ax.set_title(product)
        ax.set_xlabel(f"GOES {xray_column} at flare peak, W/m^2")
        ax.set_ylabel(index_column)
        ax.set_xscale("log")
        if len(subset) >= 2:
            corr = subset[["xray_at_flare_peak", index_column]].corr(method="spearman").iloc[0, 1]
            ax.text(0.03, 0.96, f"n={len(subset)}\nSpearman r={corr:.2f}", transform=ax.transAxes, va="top")
        else:
            ax.text(0.03, 0.96, f"n={len(subset)}", transform=ax.transAxes, va="top")

    for ax in axes_flat[len(products):]:
        ax.axis("off")

    fig.suptitle(f"{index_column} vs GOES {xray_column} at flare peak", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / f"{index_column}_vs_xray.png", dpi=160)
    plt.close(fig)


def plot_combined(stats: pd.DataFrame, index_column: str, xray_column: str, output_dir: Path) -> None:
    data = stats.dropna(subset=["xray_at_flare_peak", index_column]).copy()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for product, subset in data.groupby("product"):
        ax.scatter(subset["xray_at_flare_peak"], subset[index_column], s=58, alpha=0.82, label=product)

    ax.set_xscale("log")
    ax.set_xlabel(f"GOES {xray_column} at flare peak, W/m^2")
    ax.set_ylabel(index_column)
    ax.set_title(f"{index_column}: index vs GOES {xray_column} at flare peak")
    ax.legend(title="Product")
    fig.tight_layout()
    fig.savefig(output_dir / f"{index_column}_vs_xray_all_products.png", dpi=160)
    plt.close(fig)


def save_outputs(
    stats: pd.DataFrame,
    errors: pd.DataFrame,
    correlations: pd.DataFrame,
    output_dir: Path,
    xray_column: str,
    make_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "xray_index_peak_statistics.csv"
    errors_path = output_dir / "xray_index_peak_errors.csv"
    correlations_path = output_dir / "xray_index_peak_correlations.csv"

    stats.to_csv(stats_path, index=False)
    errors.to_csv(errors_path, index=False)
    correlations.to_csv(correlations_path, index=False)

    print(f"Saved: {stats_path}")
    print(f"Saved: {errors_path}")
    print(f"Saved: {correlations_path}")

    if make_plots:
        plt.style.use("seaborn-v0_8-whitegrid")
        for index_column in INDEX_COLUMNS:
            plot_index_vs_xray(stats, index_column, xray_column, output_dir)
        plot_combined(stats, "gsflai_index", xray_column, output_dir)
        print(f"Saved plots to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GOES X-ray vs GNSS index peak statistics.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xray-column", default="xrsb")
    parser.add_argument("--peak-time-source", choices=["event_name", "goes_max"], default="event_name")
    parser.add_argument("--max-time-delta-seconds", type=float, default=90.0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    events = load_events(results_dir)
    stats, errors = build_statistics(
        results_dir=results_dir,
        events=events,
        xray_column=args.xray_column,
        peak_time_source=args.peak_time_source,
        max_time_delta=pd.Timedelta(seconds=args.max_time_delta_seconds),
    )
    correlations = build_correlations(stats)
    save_outputs(
        stats=stats,
        errors=errors,
        correlations=correlations,
        output_dir=output_dir,
        xray_column=args.xray_column,
        make_plots=not args.no_plots,
    )

    print(f"Statistics rows: {len(stats)}")
    print(f"Errors/skips: {len(errors)}")


if __name__ == "__main__":
    main()
