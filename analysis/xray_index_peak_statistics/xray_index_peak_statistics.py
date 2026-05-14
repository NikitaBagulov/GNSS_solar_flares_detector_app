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
CLASS_SCALE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


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


def parse_flare_class(event: dict) -> tuple[str | None, float]:
    candidates = [
        str(event.get("name", "")),
        str(event.get("flare_class", "")),
        str(event.get("class", "")),
    ]
    for value in candidates:
        match = re.search(r"([ABCMX])[_\s-]?(\d+(?:[._]\d+)?)", value.upper())
        if not match:
            continue
        letter = match.group(1)
        magnitude = float(match.group(2).replace("_", "."))
        return f"{letter}{magnitude:g}", CLASS_SCALE[letter] * magnitude
    return None, np.nan


def goes_peak(goes: pd.DataFrame, xray_column: str) -> pd.Series:
    if xray_column not in goes.columns:
        raise ValueError(f"GOES CSV has no column {xray_column!r}")
    values = pd.to_numeric(goes[xray_column], errors="coerce")
    if values.dropna().empty:
        raise ValueError(f"GOES column {xray_column!r} has no numeric data")
    return goes.loc[values.idxmax()]


def goes_peak_in_window(
    goes: pd.DataFrame,
    xray_column: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.Series:
    window = goes[(goes["time"] >= start_time) & (goes["time"] <= end_time)]
    if window.empty:
        raise ValueError(f"GOES has no rows in index window {start_time}..{end_time}")
    return goes_peak(window, xray_column)


def load_event_indices(results_dir: Path, event: dict) -> dict[str, pd.DataFrame]:
    index_frames = {}
    for product in PRODUCTS:
        if not event.get("indices", {}).get(product):
            continue
        index_frames[product] = load_indices(results_dir, event, product)
    return index_frames


def index_time_window(index_frames: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    starts = []
    ends = []
    for frame in index_frames.values():
        if frame.empty:
            continue
        starts.append(frame["time"].min())
        ends.append(frame["time"].max())
    if not starts or not ends:
        return None
    return min(starts), max(ends)


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
    max_index_lag: pd.Timedelta,
    lag_step: pd.Timedelta,
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
            index_frames = load_event_indices(results_dir, event)
            flare_peak_time = parse_event_peak_time(event) if peak_time_source == "event_name" else None
            if flare_peak_time is not None and pd.notna(flare_peak_time):
                peak = nearest_row(goes, flare_peak_time, max_time_delta)
                if peak is None:
                    raise ValueError("no GOES measurement near event peak time")
                peak_source = "event_name"
            else:
                time_window = index_time_window(index_frames)
                if time_window is None:
                    raise ValueError("no index time window available for GOES peak fallback")
                peak = goes_peak_in_window(goes, xray_column, *time_window)
                flare_peak_time = peak["time"]
                peak_source = "index_window_goes_max"
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            errors.append({"event": event.get("name"), "stage": "goes", "error": str(exc)})
            continue

        for product in PRODUCTS:
            if not event.get("indices", {}).get(product):
                continue
            try:
                indices = index_frames[product]
                product_rows = []
                lag_seconds = 0.0
                while lag_seconds <= max_index_lag.total_seconds():
                    target_time = flare_peak_time + pd.Timedelta(seconds=lag_seconds)
                    nearest = nearest_row(indices, target_time, max_time_delta)
                    if nearest is not None:
                        flare_class_label, flare_class_value = parse_flare_class(event)
                        row = {
                            "event": event["name"],
                            "event_path": event["path"],
                            "flare_class": flare_class_label or event.get("class"),
                            "flare_class_value": flare_class_value,
                            "product": product,
                            "flare_peak_time": flare_peak_time,
                            "index_target_time": target_time,
                            "lag_seconds": lag_seconds,
                            "lag_minutes": lag_seconds / 60.0,
                            "goes_time": peak["time"],
                            "peak_source": peak_source,
                            "index_time": nearest["time"],
                            "index_time_delta_seconds": abs((nearest["time"] - target_time).total_seconds()),
                            "goes_time_delta_seconds": abs((peak["time"] - flare_peak_time).total_seconds()),
                            "xray_column": xray_column,
                            "xray_at_flare_peak": float(peak[xray_column]),
                        }
                        for column in INDEX_COLUMNS:
                            row[column] = float(nearest[column]) if column in nearest.index and pd.notna(nearest[column]) else np.nan
                        product_rows.append(row)
                    lag_seconds += lag_step.total_seconds()

                if not product_rows:
                    errors.append(
                        {
                            "event": event.get("name"),
                            "product": product,
                            "stage": "nearest",
                            "error": f"no index row within {max_index_lag} after event peak",
                        }
                    )
                    continue
                rows.extend(product_rows)
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
        return pd.DataFrame(columns=["product", "index", "lag_seconds", "lag_minutes", "n", "spearman_r"])

    group_columns = ["product", "lag_seconds", "lag_minutes"] if "lag_seconds" in stats.columns else ["product"]
    for group_key, product_df in stats.groupby(group_columns):
        if len(group_columns) == 3:
            product, lag_seconds, lag_minutes = group_key
        else:
            product = group_key
            lag_seconds = 0.0
            lag_minutes = 0.0
        for index_column in INDEX_COLUMNS:
            subset = product_df.dropna(subset=["xray_at_flare_peak", index_column])
            corr = np.nan
            if len(subset) >= 2:
                corr = subset[["xray_at_flare_peak", index_column]].corr(method="spearman").iloc[0, 1]
            corr_rows.append(
                {
                    "product": product,
                    "index": index_column,
                    "lag_seconds": lag_seconds,
                    "lag_minutes": lag_minutes,
                    "n": len(subset),
                    "spearman_r": corr,
                }
            )
    return pd.DataFrame(corr_rows).sort_values(["index", "product", "lag_seconds"])


def build_best_lag_correlations(correlations: pd.DataFrame) -> pd.DataFrame:
    if correlations.empty:
        return pd.DataFrame(columns=["product", "index", "lag_seconds", "lag_minutes", "n", "spearman_r"])

    rows = []
    for (product, index_column), subset in correlations.groupby(["product", "index"]):
        valid = subset.dropna(subset=["spearman_r"]).copy()
        valid = valid[valid["n"] >= 2]
        if valid.empty:
            rows.append(
                {
                    "product": product,
                    "index": index_column,
                    "lag_seconds": np.nan,
                    "lag_minutes": np.nan,
                    "n": int(subset["n"].max()) if "n" in subset else 0,
                    "spearman_r": np.nan,
                }
            )
            continue
        best = valid.loc[valid["spearman_r"].abs().idxmax()]
        rows.append(best.to_dict())
    return pd.DataFrame(rows).sort_values(["index", "product"])


def build_product_summary(stats: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "product",
        "rows",
        "events",
        "lag_slices",
        "xray_min",
        "xray_median",
        "xray_max",
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
            "lag_slices": product_df["lag_seconds"].nunique() if "lag_seconds" in product_df else 1,
            "xray_min": product_df["xray_at_flare_peak"].min(),
            "xray_median": product_df["xray_at_flare_peak"].median(),
            "xray_max": product_df["xray_at_flare_peak"].max(),
        }
        for index_column in INDEX_COLUMNS:
            row[f"{index_column}_median"] = product_df[index_column].median()
        rows.append(row)
    return pd.DataFrame(rows)[columns].sort_values("product")


def build_top_responses(stats: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if stats.empty:
        return pd.DataFrame(columns=["index", "rank", "event", "product", "flare_class", "lag_minutes", "xray_at_flare_peak", "value"])

    rows = []
    for index_column in INDEX_COLUMNS:
        subset = stats.dropna(subset=[index_column]).sort_values(index_column, ascending=False).head(top_n)
        for rank, (_, row) in enumerate(subset.iterrows(), 1):
            rows.append(
                {
                    "index": index_column,
                    "rank": rank,
                    "event": row["event"],
                    "product": row["product"],
                    "flare_class": row["flare_class"],
                    "lag_minutes": row.get("lag_minutes", 0.0),
                    "xray_at_flare_peak": row["xray_at_flare_peak"],
                    "value": row[index_column],
                }
            )
    return pd.DataFrame(rows)


def plot_index_vs_xray(stats: pd.DataFrame, index_column: str, xray_column: str, output_dir: Path) -> None:
    data = stats.dropna(subset=["xray_at_flare_peak", index_column]).copy()
    if "lag_seconds" in data.columns:
        data = data[data["lag_seconds"] == 0]
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
    if "lag_seconds" in data.columns:
        data = data[data["lag_seconds"] == 0]
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


def plot_coverage(stats: pd.DataFrame, output_dir: Path) -> None:
    if stats.empty:
        return

    counts = stats.groupby("product")["event"].nunique().reindex(PRODUCTS).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    counts.plot(kind="bar", ax=ax, color="#4c78a8")
    ax.set_title("How many events are usable for each product")
    ax.set_xlabel("Product")
    ax.set_ylabel("Events")
    ax.bar_label(ax.containers[0], padding=3)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_by_product.png", dpi=160)
    plt.close(fig)


def plot_correlation_heatmap(correlations: pd.DataFrame, output_dir: Path) -> None:
    if correlations.empty:
        return

    matrix = correlations.pivot(index="product", columns="index", values="spearman_r").reindex(PRODUCTS)
    if matrix.dropna(how="all").empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_title("Spearman correlation: GOES X-ray vs index")

    for row_idx, product in enumerate(matrix.index):
        for col_idx, index_name in enumerate(matrix.columns):
            value = matrix.loc[product, index_name]
            label = "n/a" if pd.isna(value) else f"{value:.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, label="Spearman r")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=160)
    plt.close(fig)


def plot_index_vs_flare_class(stats: pd.DataFrame, index_column: str, output_dir: Path) -> None:
    data = stats.dropna(subset=["flare_class_value", index_column]).copy()
    if "lag_seconds" in data.columns:
        data = data[data["lag_seconds"] == 0]
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for product, subset in data.groupby("product"):
        ax.scatter(subset["flare_class_value"], subset[index_column], s=58, alpha=0.82, label=product)

    ax.set_xscale("log")
    ax.set_xlabel("Flare class as GOES flux, W/m^2")
    ax.set_ylabel(index_column)
    ax.set_title(f"{index_column}: index vs flare class")
    ax.legend(title="Product")
    fig.tight_layout()
    fig.savefig(output_dir / f"{index_column}_vs_flare_class.png", dpi=160)
    plt.close(fig)


def plot_top_responses(top_responses: pd.DataFrame, output_dir: Path) -> None:
    if top_responses.empty:
        return

    for index_column, subset in top_responses.groupby("index"):
        plot_data = subset.head(10).copy()
        if plot_data.empty:
            continue
        labels = (
            plot_data["event"].astype(str)
            + " / "
            + plot_data["product"].astype(str)
            + " / +"
            + plot_data["lag_minutes"].round(1).astype(str)
            + " min"
        )
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.barh(labels[::-1], plot_data["value"].to_numpy()[::-1], color="#59a14f")
        ax.set_title(f"Top responses: {index_column}")
        ax.set_xlabel(index_column)
        fig.tight_layout()
        fig.savefig(output_dir / f"top_responses_{index_column}.png", dpi=160)
        plt.close(fig)


def plot_lag_correlations(correlations: pd.DataFrame, output_dir: Path) -> None:
    if correlations.empty or "lag_minutes" not in correlations.columns:
        return

    for index_column, index_df in correlations.groupby("index"):
        fig, ax = plt.subplots(figsize=(10, 6))
        has_lines = False
        for product, product_df in index_df.groupby("product"):
            product_df = product_df.sort_values("lag_minutes")
            valid = product_df.dropna(subset=["spearman_r"])
            if valid.empty:
                continue
            ax.plot(valid["lag_minutes"], valid["spearman_r"], marker="o", label=product)
            has_lines = True
        if not has_lines:
            plt.close(fig)
            continue
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Index lag after X-ray peak, minutes")
        ax.set_ylabel("Spearman r")
        ax.set_title(f"Lag correlation: GOES X-ray vs {index_column}")
        ax.legend(title="Product")
        fig.tight_layout()
        fig.savefig(output_dir / f"lag_correlation_{index_column}.png", dpi=160)
        plt.close(fig)


def save_outputs(
    stats: pd.DataFrame,
    errors: pd.DataFrame,
    correlations: pd.DataFrame,
    best_lag_correlations: pd.DataFrame,
    product_summary: pd.DataFrame,
    top_responses: pd.DataFrame,
    output_dir: Path,
    xray_column: str,
    make_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "xray_index_peak_statistics.csv"
    errors_path = output_dir / "xray_index_peak_errors.csv"
    correlations_path = output_dir / "xray_index_peak_correlations.csv"
    best_lag_correlations_path = output_dir / "xray_index_peak_best_lag_correlations.csv"
    product_summary_path = output_dir / "xray_index_peak_product_summary.csv"
    top_responses_path = output_dir / "xray_index_peak_top_responses.csv"

    stats.to_csv(stats_path, index=False)
    errors.to_csv(errors_path, index=False)
    correlations.to_csv(correlations_path, index=False)
    best_lag_correlations.to_csv(best_lag_correlations_path, index=False)
    product_summary.to_csv(product_summary_path, index=False)
    top_responses.to_csv(top_responses_path, index=False)

    print(f"Saved: {stats_path}")
    print(f"Saved: {errors_path}")
    print(f"Saved: {correlations_path}")
    print(f"Saved: {best_lag_correlations_path}")
    print(f"Saved: {product_summary_path}")
    print(f"Saved: {top_responses_path}")

    if make_plots:
        plt.style.use("seaborn-v0_8-whitegrid")
        plot_coverage(stats, output_dir)
        plot_correlation_heatmap(best_lag_correlations, output_dir)
        plot_lag_correlations(correlations, output_dir)
        for index_column in INDEX_COLUMNS:
            plot_index_vs_xray(stats, index_column, xray_column, output_dir)
            plot_index_vs_flare_class(stats, index_column, output_dir)
        plot_combined(stats, "gsflai_index", xray_column, output_dir)
        plot_top_responses(top_responses, output_dir)
        print(f"Saved plots to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GOES X-ray vs GNSS index peak statistics.")
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
    stats, errors = build_statistics(
        results_dir=results_dir,
        events=events,
        xray_column=args.xray_column,
        peak_time_source=args.peak_time_source,
        max_time_delta=pd.Timedelta(seconds=args.max_time_delta_seconds),
        max_index_lag=pd.Timedelta(minutes=args.max_index_lag_minutes),
        lag_step=pd.Timedelta(seconds=args.lag_step_seconds),
    )
    correlations = build_correlations(stats)
    best_lag_correlations = build_best_lag_correlations(correlations)
    product_summary = build_product_summary(stats)
    top_responses = build_top_responses(stats)
    save_outputs(
        stats=stats,
        errors=errors,
        correlations=correlations,
        best_lag_correlations=best_lag_correlations,
        product_summary=product_summary,
        top_responses=top_responses,
        output_dir=output_dir,
        xray_column=args.xray_column,
        make_plots=not args.no_plots,
    )

    print(f"Statistics rows: {len(stats)}")
    print(f"Errors/skips: {len(errors)}")
    if not product_summary.empty:
        print("\nRows by product:")
        print(product_summary[["product", "events", "lag_slices", "rows"]].to_string(index=False))
    if not best_lag_correlations.empty:
        print("\nBest lag correlations:")
        print(best_lag_correlations[["product", "index", "lag_minutes", "n", "spearman_r"]].to_string(index=False))


if __name__ == "__main__":
    main()
