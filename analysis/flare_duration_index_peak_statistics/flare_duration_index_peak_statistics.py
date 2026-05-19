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
DEFAULT_MAX_DURATION_MINUTES = 120.0

sys.path.insert(0, str(XRAY_STATS_DIR))
from xray_index_peak_statistics import (  # noqa: E402
    FLARE_CLASS_MARKERS,
    INDEX_COLUMNS,
    PLOTTED_FLARE_CLASSES,
    PRODUCTS,
    build_statistics,
    flare_class_letter,
    load_events,
    load_goes,
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


def estimate_goes_half_peak_duration(
    results_dir: Path,
    event: dict,
    xray_column: str,
    peak_time: pd.Timestamp,
    peak_value: float,
) -> dict | None:
    if pd.isna(peak_time) or pd.isna(peak_value) or peak_value <= 0:
        return None

    goes = load_goes(results_dir, event, xray_column)
    goes = goes.dropna(subset=["time", xray_column]).sort_values("time").reset_index(drop=True)
    if goes.empty:
        return None

    nearest_idx = (goes["time"] - peak_time).abs().idxmin()
    threshold = peak_value * 0.5
    above_threshold = goes[xray_column] >= threshold
    if not bool(above_threshold.iloc[nearest_idx]):
        nearest_idx = int(goes[xray_column].idxmax())
        if not bool(above_threshold.iloc[nearest_idx]):
            return None

    left_idx = nearest_idx
    while left_idx > 0 and bool(above_threshold.iloc[left_idx - 1]):
        left_idx -= 1

    right_idx = nearest_idx
    while right_idx < len(goes) - 1 and bool(above_threshold.iloc[right_idx + 1]):
        right_idx += 1

    start_time = goes.loc[left_idx, "time"]
    end_time = goes.loc[right_idx, "time"]
    duration_seconds = (end_time - start_time).total_seconds()
    if duration_seconds <= 0 and len(goes) > 1:
        cadence = goes["time"].diff().dt.total_seconds().dropna().median()
        duration_seconds = float(cadence) if pd.notna(cadence) and cadence > 0 else 0.0
        end_time = start_time + pd.Timedelta(seconds=duration_seconds)
    if duration_seconds <= 0:
        return None

    return {
        "flare_start_time": start_time,
        "flare_peak_time_from_name": pd.NaT,
        "flare_end_time": end_time,
        "flare_duration_seconds": duration_seconds,
        "flare_duration_minutes": duration_seconds / 60.0,
        "flare_duration_source": "goes_half_peak_width",
    }


def add_flare_duration(
    stats: pd.DataFrame,
    events: list[dict],
    results_dir: Path,
    xray_column: str,
    max_duration_minutes: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    errors = []
    duration_by_event = {}
    event_by_name = {event["name"]: event for event in events}

    for event in events:
        parsed = parse_event_times(event)
        if parsed is not None:
            start_time, peak_time, end_time = parsed
            duration_by_event[event["name"]] = {
                "flare_start_time": start_time,
                "flare_peak_time_from_name": peak_time,
                "flare_end_time": end_time,
                "flare_duration_seconds": (end_time - start_time).total_seconds(),
                "flare_duration_minutes": (end_time - start_time).total_seconds() / 60.0,
                "flare_duration_source": "event_name",
            }

    for _, row in stats.iterrows():
        duration = duration_by_event.get(row["event"])
        if duration is None:
            event = event_by_name.get(row["event"])
            if event is None:
                errors.append({"event": row.get("event"), "stage": "duration", "error": "event not found"})
                continue
            try:
                duration = estimate_goes_half_peak_duration(
                    results_dir=results_dir,
                    event=event,
                    xray_column=xray_column,
                    peak_time=row["flare_peak_time"],
                    peak_value=float(row["xray_at_flare_peak"]),
                )
            except (ValueError, OSError, pd.errors.ParserError) as exc:
                errors.append({"event": row.get("event"), "stage": "duration", "error": str(exc)})
                continue
            if duration is None:
                errors.append(
                    {
                        "event": row.get("event"),
                        "stage": "duration",
                        "error": "could not parse duration from name or estimate GOES half-peak width",
                    }
                )
                continue
            duration_by_event[row["event"]] = duration
        if duration["flare_duration_minutes"] > max_duration_minutes:
            errors.append(
                {
                    "event": row.get("event"),
                    "stage": "duration",
                    "error": (
                        f"duration {duration['flare_duration_minutes']:.2f} min exceeds "
                        f"max {max_duration_minutes:.2f} min"
                    ),
                }
            )
            continue
        output_row = row.to_dict()
        output_row.update(duration)
        rows.append(output_row)

    if rows:
        stats_with_duration = pd.DataFrame(rows)
    else:
        extra_columns = [
            "flare_start_time",
            "flare_peak_time_from_name",
            "flare_end_time",
            "flare_duration_seconds",
            "flare_duration_minutes",
            "flare_duration_source",
        ]
        stats_with_duration = pd.DataFrame(columns=[*stats.columns, *extra_columns])
    return stats_with_duration, pd.DataFrame(errors)


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
    if stats.empty or "product" not in stats.columns:
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
    required_columns = {"flare_duration_minutes", index_column, "flare_class"}
    if stats.empty or not required_columns.issubset(stats.columns):
        print(f"No C/M/X duration plot data for {index_column}")
        return

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
        median_duration = subset["flare_duration_minutes"].median()
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
        if pd.notna(median_duration):
            ax.axvline(
                median_duration,
                color="black",
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
                label=f"median {median_duration:.1f} min",
            )
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
    parser.add_argument("--max-duration-minutes", type=float, default=DEFAULT_MAX_DURATION_MINUTES)
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
    stats, duration_errors = add_flare_duration(
        stats=xray_stats,
        events=events,
        results_dir=results_dir,
        xray_column=args.xray_column,
        max_duration_minutes=args.max_duration_minutes,
    )
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
