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
XRAY_STATS_DIR = REPO_ROOT / "analysis" / "xray_index_peak_statistics"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

sys.path.insert(0, str(XRAY_STATS_DIR))
from xray_index_peak_statistics import (  # noqa: E402
    FLARE_CLASS_MARKERS,
    INDEX_COLUMNS,
    PLOTTED_FLARE_CLASSES,
    PRODUCTS,
    build_statistics,
    flare_class_letter,
    load_events,
)


def parse_event_times(event: dict) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp] | None:
    timestamps = re.findall(r"\d{8}T\d{6}", event.get("name", ""))
    if len(timestamps) < 3:
        return None
    start_time, peak_time, end_time = [
        pd.to_datetime(value, format="%Y%m%dT%H%M%S", utc=True, errors="coerce")
        for value in timestamps[:3]
    ]
    if pd.isna(start_time) or pd.isna(peak_time) or pd.isna(end_time):
        return None
    if end_time <= start_time:
        return None
    return start_time, peak_time, end_time


def add_flare_duration(stats: pd.DataFrame, events: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    errors = []
    duration_by_event = {}

    for event in events:
        parsed = parse_event_times(event)
        if parsed is None:
            errors.append(
                {
                    "event": event.get("name"),
                    "stage": "duration",
                    "error": "could not parse valid start/peak/end timestamps from event name",
                }
            )
            continue
        start_time, peak_time, end_time = parsed
        duration_by_event[event["name"]] = {
            "flare_start_time": start_time,
            "flare_peak_time_from_name": peak_time,
            "flare_end_time": end_time,
            "flare_duration_seconds": (end_time - start_time).total_seconds(),
            "flare_duration_minutes": (end_time - start_time).total_seconds() / 60.0,
        }

    for _, row in stats.iterrows():
        duration = duration_by_event.get(row["event"])
        if duration is None:
            continue
        output_row = row.to_dict()
        output_row.update(duration)
        rows.append(output_row)

    return pd.DataFrame(rows), pd.DataFrame(errors)


def build_duration_correlations(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = ["product", "index", "lag_seconds", "lag_minutes", "n", "spearman_r"]
    if stats.empty:
        return pd.DataFrame(columns=columns)

    data = stats.dropna(subset=["flare_duration_minutes"]).copy()
    group_columns = ["product", "lag_seconds", "lag_minutes"]
    for (product, lag_seconds, lag_minutes), product_df in data.groupby(group_columns):
        for index_column in INDEX_COLUMNS:
            subset = product_df.dropna(subset=["flare_duration_minutes", index_column])
            corr = np.nan
            if len(subset) >= 2:
                corr = subset[["flare_duration_minutes", index_column]].corr(method="spearman").iloc[0, 1]
            rows.append(
                {
                    "product": product,
                    "index": index_column,
                    "lag_seconds": lag_seconds,
                    "lag_minutes": lag_minutes,
                    "n": len(subset),
                    "spearman_r": corr,
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(["index", "product", "lag_seconds"])


def build_product_summary(stats: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "product",
        "rows",
        "events",
        "lag_slices",
        "duration_min",
        "duration_median",
        "duration_max",
        "day_night_index_median",
        "gsflai_index_median",
        "isfai_index_median",
    ]
    if stats.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for product, product_df in stats.groupby("product"):
        row = {
            "product": product,
            "rows": len(product_df),
            "events": product_df["event"].nunique(),
            "lag_slices": product_df["lag_seconds"].nunique(),
            "duration_min": product_df["flare_duration_minutes"].min(),
            "duration_median": product_df["flare_duration_minutes"].median(),
            "duration_max": product_df["flare_duration_minutes"].max(),
        }
        for index_column in INDEX_COLUMNS:
            row[f"{index_column}_median"] = product_df[index_column].median()
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values("product")


def plot_index_vs_duration_by_flare_class(stats: pd.DataFrame, index_column: str, output_dir: Path) -> None:
    data = stats.dropna(subset=["flare_duration_minutes", index_column, "flare_class"]).copy()
    if "lag_seconds" in data.columns:
        data = data[data["lag_seconds"] == 0]
    data = data[data["flare_duration_minutes"] > 0]
    data["flare_class_letter"] = data["flare_class"].map(flare_class_letter)
    data = data[data["flare_class_letter"].isin(PLOTTED_FLARE_CLASSES)]
    if data.empty:
        print(f"No C/M/X duration plot data for {index_column}")
        return

    products = [product for product in PRODUCTS if product in set(data["product"])]
    cols = 2
    rows_count = math.ceil(len(products) / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(13, 4.8 * rows_count), squeeze=False)
    axes_flat = axes.ravel()

    for ax, product in zip(axes_flat, products):
        subset = data[data["product"] == product]
        for flare_class in PLOTTED_FLARE_CLASSES:
            class_subset = subset[subset["flare_class_letter"] == flare_class]
            if class_subset.empty:
                continue
            ax.scatter(
                class_subset["flare_duration_minutes"],
                class_subset[index_column],
                s=64,
                alpha=0.82,
                marker=FLARE_CLASS_MARKERS[flare_class],
                label=f"{flare_class}-class",
            )

        ax.set_title(product)
        ax.set_xlabel("Flare duration, minutes")
        ax.set_ylabel(index_column)
        counts = subset["flare_class_letter"].value_counts().reindex(PLOTTED_FLARE_CLASSES).fillna(0).astype(int)
        ax.text(
            0.03,
            0.96,
            "n=" + str(len(subset)) + "\n" + ", ".join(f"{name}:{counts[name]}" for name in PLOTTED_FLARE_CLASSES),
            transform=ax.transAxes,
            va="top",
        )
        ax.legend(title="Flare class")

    for ax in axes_flat[len(products):]:
        ax.axis("off")

    fig.suptitle(f"{index_column} at X-ray peak vs flare duration by flare class", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / f"{index_column}_vs_flare_duration_by_flare_class.png", dpi=160)
    plt.close(fig)


def save_outputs(
    stats: pd.DataFrame,
    errors: pd.DataFrame,
    correlations: pd.DataFrame,
    product_summary: pd.DataFrame,
    output_dir: Path,
    make_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "flare_duration_index_peak_statistics.csv"
    errors_path = output_dir / "flare_duration_index_peak_errors.csv"
    correlations_path = output_dir / "flare_duration_index_peak_correlations.csv"
    product_summary_path = output_dir / "flare_duration_index_peak_product_summary.csv"

    stats.to_csv(stats_path, index=False)
    errors.to_csv(errors_path, index=False)
    correlations.to_csv(correlations_path, index=False)
    product_summary.to_csv(product_summary_path, index=False)

    print(f"Saved: {stats_path}")
    print(f"Saved: {errors_path}")
    print(f"Saved: {correlations_path}")
    print(f"Saved: {product_summary_path}")

    if make_plots:
        plt.style.use("seaborn-v0_8-whitegrid")
        for index_column in INDEX_COLUMNS:
            plot_index_vs_duration_by_flare_class(stats, index_column, output_dir)
        print(f"Saved plots to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build flare duration vs GNSS index peak statistics.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xray-column", default="xrsb")
    parser.add_argument("--peak-time-source", choices=["event_name", "goes_max"], default="event_name")
    parser.add_argument("--max-time-delta-seconds", type=float, default=90.0)
    parser.add_argument("--max-index-lag-minutes", type=float, default=10.0)
    parser.add_argument("--lag-step-seconds", type=float, default=60.0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    events = load_events(results_dir)
    xray_stats, xray_errors = build_statistics(
        results_dir=results_dir,
        events=events,
        xray_column=args.xray_column,
        peak_time_source=args.peak_time_source,
        max_time_delta=pd.Timedelta(seconds=args.max_time_delta_seconds),
        max_index_lag=pd.Timedelta(minutes=args.max_index_lag_minutes),
        lag_step=pd.Timedelta(seconds=args.lag_step_seconds),
    )
    stats, duration_errors = add_flare_duration(xray_stats, events)
    errors = pd.concat([xray_errors, duration_errors], ignore_index=True)
    correlations = build_duration_correlations(stats)
    product_summary = build_product_summary(stats)
    save_outputs(
        stats=stats,
        errors=errors,
        correlations=correlations,
        product_summary=product_summary,
        output_dir=output_dir,
        make_plots=not args.no_plots,
    )

    print(f"Statistics rows: {len(stats)}")
    print(f"Errors/skips: {len(errors)}")
    if not product_summary.empty:
        print("\nRows by product:")
        print(product_summary[["product", "events", "lag_slices", "rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
