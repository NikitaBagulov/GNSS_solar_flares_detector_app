from __future__ import annotations

import argparse
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
SOLAR_DRIVERS = ("xray", "euv", "euv_derivative")
DRIVER_COLORS = {
    "xray": "#4c78a8",
    "euv": "#f58518",
    "euv_derivative": "#54a24b",
}
DRIVER_LABELS = {
    "xray": "X-ray peak",
    "euv": "EUV peak",
    "euv_derivative": "dEUV/dt peak",
}


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


def parse_event_times(event: dict) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    timestamps = re.findall(r"\d{8}T\d{6}", event.get("name", ""))
    if len(timestamps) < 3:
        return None, None, None
    parsed = [
        pd.to_datetime(value, format="%Y%m%dT%H%M%S", utc=True, errors="coerce")
        for value in timestamps[:3]
    ]
    if any(pd.isna(value) for value in parsed):
        return None, None, None
    return parsed[0], parsed[1], parsed[2]


def load_goes(results_dir: Path, event: dict, xray_column: str) -> pd.DataFrame:
    path = event_file_path(results_dir, event, "goes_xray", "goes_xray.csv")
    df = normalize_time_column(pd.read_csv(path), preferred="time")
    if xray_column not in df.columns:
        raise ValueError(f"GOES CSV has no column {xray_column!r}")
    df[xray_column] = pd.to_numeric(df[xray_column], errors="coerce")
    return df.dropna(subset=[xray_column])


def load_soho_sem(results_dir: Path, event: dict, euv_column: str) -> pd.DataFrame:
    path = event_file_path(results_dir, event, "soho_sem", "soho_sem.csv")
    df = normalize_time_column(pd.read_csv(path), preferred="time")
    if euv_column not in df.columns:
        raise ValueError(f"SOHO SEM CSV has no column {euv_column!r}")
    df[euv_column] = pd.to_numeric(df[euv_column], errors="coerce")
    return df.dropna(subset=[euv_column])


def load_indices(results_dir: Path, event: dict, product: str) -> pd.DataFrame:
    path = event_file_path(results_dir, event, "indices", f"indices_{product}.csv")
    df = normalize_time_column(pd.read_csv(path), preferred="time")
    for column in INDEX_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def clip_window(
    frame: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    padding: pd.Timedelta,
) -> pd.DataFrame:
    return frame[(frame["time"] >= start_time - padding) & (frame["time"] <= end_time + padding)]


def peak_row(frame: pd.DataFrame, value_column: str) -> pd.Series:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    if values.dropna().empty:
        raise ValueError(f"No numeric data in {value_column!r}")
    return frame.loc[values.idxmax()]


def add_derivative(frame: pd.DataFrame, value_column: str, derivative_column: str) -> pd.DataFrame:
    frame = frame.sort_values("time").copy()
    seconds = frame["time"].astype("int64") / 1e9
    values = pd.to_numeric(frame[value_column], errors="coerce")
    frame[derivative_column] = values.diff() / pd.Series(seconds, index=frame.index).diff()
    return frame.dropna(subset=[derivative_column])


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


def load_event_indices(results_dir: Path, event: dict) -> dict[str, pd.DataFrame]:
    index_frames = {}
    for product in PRODUCTS:
        if not event.get("indices", {}).get(product):
            continue
        index_frames[product] = load_indices(results_dir, event, product)
    return index_frames


def driver_peaks(
    results_dir: Path,
    event: dict,
    index_frames: dict[str, pd.DataFrame],
    xray_column: str,
    euv_column: str,
    window_padding: pd.Timedelta,
) -> dict[str, dict]:
    event_start, _, event_end = parse_event_times(event)
    if event_start is None or event_end is None:
        index_window = index_time_window(index_frames)
        if index_window is None:
            raise ValueError("Cannot infer event window")
        event_start, event_end = index_window

    peaks = {}

    goes = clip_window(load_goes(results_dir, event, xray_column), event_start, event_end, window_padding)
    if not goes.empty:
        row = peak_row(goes, xray_column)
        peaks["xray"] = {"time": row["time"], "value": float(row[xray_column])}

    sem = clip_window(load_soho_sem(results_dir, event, euv_column), event_start, event_end, window_padding)
    if not sem.empty:
        row = peak_row(sem, euv_column)
        peaks["euv"] = {"time": row["time"], "value": float(row[euv_column])}
        derivative = add_derivative(sem, euv_column, "euv_derivative")
        if not derivative.empty:
            row = peak_row(derivative, "euv_derivative")
            peaks["euv_derivative"] = {"time": row["time"], "value": float(row["euv_derivative"])}

    return peaks


def build_lag_rows(
    results_dir: Path,
    events: list[dict],
    xray_column: str,
    euv_column: str,
    window_padding: pd.Timedelta,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    errors = []
    usable_events = [
        event
        for event in events
        if event.get("sources", {}).get("goes_xray")
        and event.get("sources", {}).get("soho_sem")
        and any(event.get("indices", {}).values())
    ]

    print(f"Events in results: {len(events)}")
    print(f"Events with GOES, SOHO SEM, and at least one index file: {len(usable_events)}")

    for event_idx, event in enumerate(usable_events, 1):
        if event_idx == 1 or event_idx % 10 == 0 or event_idx == len(usable_events):
            print(f"[{event_idx}/{len(usable_events)}] {event.get('name')}")

        try:
            index_frames = load_event_indices(results_dir, event)
            peaks = driver_peaks(
                results_dir=results_dir,
                event=event,
                index_frames=index_frames,
                xray_column=xray_column,
                euv_column=euv_column,
                window_padding=window_padding,
            )
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            errors.append({"event": event.get("name"), "stage": "solar_drivers", "error": str(exc)})
            continue

        for product, index_frame in index_frames.items():
            for index_column in INDEX_COLUMNS:
                if index_column not in index_frame.columns:
                    continue
                try:
                    index_peak = peak_row(index_frame.dropna(subset=[index_column]), index_column)
                except ValueError as exc:
                    errors.append(
                        {
                            "event": event.get("name"),
                            "product": product,
                            "index": index_column,
                            "stage": "index_peak",
                            "error": str(exc),
                        }
                    )
                    continue

                for driver_name in SOLAR_DRIVERS:
                    peak = peaks.get(driver_name)
                    if not peak:
                        errors.append(
                            {
                                "event": event.get("name"),
                                "product": product,
                                "index": index_column,
                                "stage": driver_name,
                                "error": "driver peak unavailable",
                            }
                        )
                        continue
                    lag_seconds = (index_peak["time"] - peak["time"]).total_seconds()
                    rows.append(
                        {
                            "event": event["name"],
                            "event_path": event["path"],
                            "product": product,
                            "index": index_column,
                            "driver": driver_name,
                            "driver_peak_time": peak["time"],
                            "driver_peak_value": peak["value"],
                            "index_peak_time": index_peak["time"],
                            "index_peak_value": float(index_peak[index_column]),
                            "lag_seconds": lag_seconds,
                            "lag_minutes": lag_seconds / 60.0,
                        }
                    )

    return pd.DataFrame(rows), pd.DataFrame(errors)


def build_summary(lags: pd.DataFrame) -> pd.DataFrame:
    if lags.empty:
        return pd.DataFrame(
            columns=[
                "driver",
                "product",
                "index",
                "n",
                "lag_minutes_median",
                "lag_minutes_mean",
                "lag_minutes_min",
                "lag_minutes_max",
            ]
        )

    return (
        lags.groupby(["driver", "product", "index"])["lag_minutes"]
        .agg(
            n="count",
            lag_minutes_median="median",
            lag_minutes_mean="mean",
            lag_minutes_min="min",
            lag_minutes_max="max",
        )
        .reset_index()
        .sort_values(["driver", "index", "product"])
    )


def plot_lag_points(lags: pd.DataFrame, output_dir: Path) -> None:
    if lags.empty:
        return

    for driver, driver_df in lags.groupby("driver"):
        fig, axes = plt.subplots(len(INDEX_COLUMNS), 1, figsize=(12, 4.2 * len(INDEX_COLUMNS)), squeeze=False)
        for ax, index_column in zip(axes.ravel(), INDEX_COLUMNS):
            subset = driver_df[driver_df["index"] == index_column]
            if subset.empty:
                ax.axis("off")
                continue

            y_positions = {product: idx for idx, product in enumerate(PRODUCTS)}
            for product in PRODUCTS:
                product_df = subset[subset["product"] == product].dropna(subset=["lag_minutes"])
                if product_df.empty:
                    continue
                y = np.full(len(product_df), y_positions[product], dtype=float)
                if len(product_df) > 1:
                    y += np.linspace(-0.12, 0.12, len(product_df))
                ax.scatter(product_df["lag_minutes"], y, s=54, alpha=0.82, label=product)
                median = product_df["lag_minutes"].median()
                ax.plot([median, median], [y_positions[product] - 0.25, y_positions[product] + 0.25], color="black", linewidth=2)

            ax.axvline(0, color="black", linewidth=0.9, alpha=0.55)
            ax.set_yticks(range(len(PRODUCTS)), PRODUCTS)
            ax.set_title(f"{driver}: each event lag to {index_column} peak")
            ax.set_xlabel("Index peak time - driver peak time, minutes")
            ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"lag_points_{driver}.png", dpi=160)
        plt.close(fig)


def plot_median_lag_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return

    for driver, driver_df in summary.groupby("driver"):
        for index_column, index_df in driver_df.groupby("index"):
            plot_df = index_df.set_index("product").reindex(PRODUCTS).dropna(subset=["lag_minutes_median"])
            if plot_df.empty:
                continue
            colors = ["#4c78a8" if value >= 0 else "#e15759" for value in plot_df["lag_minutes_median"]]
            fig, ax = plt.subplots(figsize=(9, 4.8))
            bars = ax.barh(plot_df.index, plot_df["lag_minutes_median"], color=colors, alpha=0.88)
            ax.axvline(0, color="black", linewidth=0.9)
            ax.set_title(f"Median lag: {driver} -> {index_column}")
            ax.set_xlabel("Index peak time - driver peak time, minutes")
            for bar, (_, row) in zip(bars, plot_df.iterrows()):
                value = row["lag_minutes_median"]
                label = f"{value:.1f} min, n={int(row['n'])}"
                x = value + (0.2 if value >= 0 else -0.2)
                ha = "left" if value >= 0 else "right"
                ax.text(x, bar.get_y() + bar.get_height() / 2, label, va="center", ha=ha, fontsize=9)
            fig.tight_layout()
            fig.savefig(output_dir / f"median_lag_bar_{driver}_{index_column}.png", dpi=160)
            plt.close(fig)


def plot_product_driver_comparison(lags: pd.DataFrame, output_dir: Path) -> None:
    if lags.empty:
        return

    driver_offsets = {
        "xray": -0.18,
        "euv": 0.0,
        "euv_derivative": 0.18,
    }
    index_positions = {index_name: idx for idx, index_name in enumerate(INDEX_COLUMNS)}

    for product, product_df in lags.groupby("product"):
        fig, ax = plt.subplots(figsize=(12, 5.8))
        has_points = False
        for driver in SOLAR_DRIVERS:
            driver_df = product_df[product_df["driver"] == driver]
            if driver_df.empty:
                continue
            for index_name in INDEX_COLUMNS:
                subset = driver_df[driver_df["index"] == index_name].dropna(subset=["lag_minutes"])
                if subset.empty:
                    continue
                y_base = index_positions[index_name] + driver_offsets[driver]
                y = np.full(len(subset), y_base, dtype=float)
                if len(subset) > 1:
                    y += np.linspace(-0.035, 0.035, len(subset))
                ax.scatter(
                    subset["lag_minutes"],
                    y,
                    s=58,
                    alpha=0.82,
                    color=DRIVER_COLORS[driver],
                    label=DRIVER_LABELS[driver] if not has_points or driver not in ax.get_legend_handles_labels()[1] else None,
                )
                median = subset["lag_minutes"].median()
                ax.plot(
                    [median, median],
                    [y_base - 0.07, y_base + 0.07],
                    color=DRIVER_COLORS[driver],
                    linewidth=3,
                )
                has_points = True

        if not has_points:
            plt.close(fig)
            continue

        ax.axvline(0, color="black", linewidth=0.9, alpha=0.55)
        ax.set_yticks(range(len(INDEX_COLUMNS)), INDEX_COLUMNS)
        ax.set_xlabel("Index peak time - solar driver peak time, minutes")
        ax.set_title(f"{product}: lag comparison across solar drivers")
        ax.grid(True, axis="x", alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for handle, label in zip(handles, labels):
            if label:
                unique[label] = handle
        ax.legend(unique.values(), unique.keys(), title="Solar driver", loc="best")

        fig.tight_layout()
        fig.savefig(output_dir / f"lag_points_by_driver_{product}.png", dpi=160)
        plt.close(fig)


def save_summary_table_image(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return

    table = summary.copy()
    table["lag_minutes_median"] = table["lag_minutes_median"].round(2)
    table["lag_minutes_mean"] = table["lag_minutes_mean"].round(2)
    table = table[["driver", "product", "index", "n", "lag_minutes_median", "lag_minutes_mean"]]

    fig_height = max(4, min(24, 0.35 * len(table) + 1.5))
    fig, ax = plt.subplots(figsize=(13, fig_height))
    ax.axis("off")
    rendered = ax.table(
        cellText=table.astype(str).values,
        colLabels=["driver", "product", "index", "n", "median lag, min", "mean lag, min"],
        loc="center",
        cellLoc="center",
    )
    rendered.auto_set_font_size(False)
    rendered.set_fontsize(8.5)
    rendered.scale(1, 1.25)
    ax.set_title("Solar driver -> GNSS index lag summary", pad=16)
    fig.tight_layout()
    fig.savefig(output_dir / "lag_summary_table.png", dpi=180)
    plt.close(fig)


def save_outputs(lags: pd.DataFrame, errors: pd.DataFrame, summary: pd.DataFrame, output_dir: Path, make_plots: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lags_path = output_dir / "solar_driver_index_lags.csv"
    errors_path = output_dir / "solar_driver_index_lag_errors.csv"
    summary_path = output_dir / "solar_driver_index_lag_summary.csv"

    lags.to_csv(lags_path, index=False)
    errors.to_csv(errors_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {lags_path}")
    print(f"Saved: {errors_path}")
    print(f"Saved: {summary_path}")

    if make_plots:
        plt.style.use("seaborn-v0_8-whitegrid")
        plot_lag_points(lags, output_dir)
        plot_product_driver_comparison(lags, output_dir)
        plot_median_lag_bars(summary, output_dir)
        save_summary_table_image(summary, output_dir)
        print(f"Saved plots to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure lags between solar driver peaks and GNSS index peaks.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xray-column", default="xrsb")
    parser.add_argument("--euv-column", default="flux_01_50")
    parser.add_argument("--window-padding-minutes", type=float, default=10.0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    events = load_events(results_dir)
    lags, errors = build_lag_rows(
        results_dir=results_dir,
        events=events,
        xray_column=args.xray_column,
        euv_column=args.euv_column,
        window_padding=pd.Timedelta(minutes=args.window_padding_minutes),
    )
    summary = build_summary(lags)
    save_outputs(lags, errors, summary, output_dir, make_plots=not args.no_plots)

    print(f"Lag rows: {len(lags)}")
    print(f"Errors/skips: {len(errors)}")
    if not summary.empty:
        print("\nMedian lag summary:")
        print(summary[["driver", "product", "index", "n", "lag_minutes_median"]].to_string(index=False))


if __name__ == "__main__":
    main()
