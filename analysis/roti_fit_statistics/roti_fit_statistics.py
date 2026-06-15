from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
R_EARTH = 6371.0
MIN_POINTS = 10
PLOT_SAMPLE_SIZE = 100_000

SLICE_COLUMNS = [
    "event",
    "map_path",
    "time",
    "xrsb",
    "A",
    "B",
    "n_points",
    "actual_mean",
    "actual_std",
    "predicted_mean",
    "predicted_std",
    "mae",
    "rmse",
    "r2",
    "pearson_actual_predicted",
]


def parse_time(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def subsolar_point(dt_value: pd.Timestamp) -> tuple[float, float]:
    dt_value = dt_value.tz_convert(None).to_pydatetime()
    year, month = dt_value.year, dt_value.month
    day = dt_value.day + (
        dt_value.hour + (dt_value.minute + dt_value.second / 60.0) / 60.0
    ) / 24.0
    if month <= 2:
        year -= 1
        month += 12
    century = year // 100
    gregorian = 2 - century + century // 4
    jd = (
        int(365.25 * (year + 4716))
        + int(30.6001 * (month + 1))
        + day
        + gregorian
        - 1524.5
    )
    t = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + 36000.76983 * t + 0.0003032 * t * t) % 360.0
    mean_anomaly = (357.52911 + 35999.05029 * t - 0.0001537 * t * t) % 360.0
    anomaly_rad = math.radians(mean_anomaly)
    correction = (
        (1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(anomaly_rad)
        + (0.019993 - 0.000101 * t) * math.sin(2 * anomaly_rad)
        + 0.000289 * math.sin(3 * anomaly_rad)
    )
    true_longitude = l0 + correction
    epsilon = math.radians(23.439291 - 0.0130042 * t)
    longitude_rad = math.radians(true_longitude)
    declination = math.asin(math.sin(epsilon) * math.sin(longitude_rad))
    right_ascension = math.atan2(
        math.cos(epsilon) * math.sin(longitude_rad), math.cos(longitude_rad)
    )
    right_ascension_deg = (math.degrees(right_ascension) + 360.0) % 360.0
    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - (t * t * t) / 38710000.0
    ) % 360.0
    gha = (gmst - right_ascension_deg + 360.0) % 360.0
    return math.degrees(declination), -gha % 360.0 - 180.0


def day_geometry(
    lat: np.ndarray, lon: np.ndarray, sub_lat: float, sub_lon: float
) -> np.ndarray:
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    sub_lat_rad = math.radians(sub_lat)
    sub_lon_rad = math.radians(sub_lon)
    delta_lon = (lon_rad - sub_lon_rad + np.pi) % (2 * np.pi) - np.pi
    cos_chi = (
        np.sin(lat_rad) * math.sin(sub_lat_rad)
        + np.cos(lat_rad) * math.cos(sub_lat_rad) * np.cos(delta_lon)
    )
    return np.clip(cos_chi, -1.0, 1.0)


def distance_weight(cos_chi: np.ndarray) -> np.ndarray:
    dist = R_EARTH * np.arccos(np.clip(cos_chi, -1.0, 1.0))
    return 1.0 / (1.0 + (dist / R_EARTH) ** 2)


def read_map_node(node: h5py.Group | h5py.Dataset) -> tuple[np.ndarray, ...]:
    if isinstance(node, h5py.Group):
        return tuple(np.asarray(node[name][:], dtype=float) for name in ("lat", "lon", "vals"))
    data = node[:]
    return tuple(np.asarray(data[name], dtype=float) for name in ("lat", "lon", "vals"))


def safe_corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, method: str) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 2 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return np.nan
    return float(frame.corr(method=method).iloc[0, 1])


def fit_slice(node: h5py.Group | h5py.Dataset, time: pd.Timestamp) -> dict | None:
    lat, lon, values = read_map_node(node)
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(values)
    lat, lon, values = lat[valid], lon[valid], values[valid]
    sub_lat, sub_lon = subsolar_point(time)
    cos_chi = day_geometry(lat, lon, sub_lat, sub_lon)
    day = cos_chi > 0
    x = cos_chi[day]
    actual = values[day]
    if len(x) < MIN_POINTS:
        return None

    weights = distance_weight(x)
    sw = np.sum(weights)
    swx = np.sum(weights * x)
    swx2 = np.sum(weights * x**2)
    swy = np.sum(weights * actual)
    swxy = np.sum(weights * x * actual)
    determinant = sw * swx2 - swx**2
    if not np.isfinite(determinant) or abs(determinant) < np.finfo(float).eps:
        return None

    a_value = (sw * swxy - swx * swy) / determinant
    b_value = (swx2 * swy - swx * swxy) / determinant
    predicted = a_value * x + b_value
    residuals = actual - predicted
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1.0 - np.sum(residuals**2) / denominator if denominator > 0 else np.nan
    return {
        "A": float(a_value),
        "B": float(b_value),
        "n_points": len(actual),
        "actual_mean": float(np.mean(actual)),
        "actual_std": float(np.std(actual)),
        "predicted_mean": float(np.mean(predicted)),
        "predicted_std": float(np.std(predicted)),
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "r2": float(r2),
        "pearson_actual_predicted": safe_corr(actual, predicted, "pearson"),
        "actual": actual,
        "predicted": predicted,
    }


def load_xray(event_dir: Path) -> pd.DataFrame:
    path = event_dir / "goes_xray" / "goes_xray.csv"
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame["xrsb"] = pd.to_numeric(frame["xrsb"], errors="coerce")
    return frame.dropna(subset=["xrsb"]).sort_index()


def load_events(results_dir: Path) -> list[dict]:
    sys.path.insert(0, str(REPO_ROOT))
    from results_server import scan_events

    return scan_events(results_dir)


def event_file_path(results_dir: Path, event: dict, *parts: str) -> Path:
    return results_dir / event["path"] / Path(*parts)


def interpolate_xray(xray: pd.DataFrame, times: list[pd.Timestamp]) -> np.ndarray:
    x = xray.index.asi8.astype(float)
    targets = pd.DatetimeIndex(times).asi8.astype(float)
    return np.interp(targets, x, xray["xrsb"].to_numpy(dtype=float), left=np.nan, right=np.nan)


def select_usable_events(results_dir: Path, events: list[dict]) -> list[dict]:
    usable = []
    for event in events:
        if not event.get("maps", {}).get("roti") or not event.get("sources", {}).get("goes_xray"):
            continue
        usable.append(
            {
                **event,
                "event": event["name"],
                "event_dir": results_dir / event["path"],
                "map_path": event_file_path(results_dir, event, "maps", "map_roti.h5"),
            }
        )
    return usable


def parse_peak_time(event_name: str) -> pd.Timestamp | None:
    timestamps = re.findall(r"\d{8}T\d{6}", event_name)
    if len(timestamps) < 2:
        return None
    return pd.to_datetime(timestamps[1], format="%Y%m%dT%H%M%S", utc=True, errors="coerce")


def add_correlation_rows(rows: list[dict], frame: pd.DataFrame, scope: str) -> None:
    for coefficient in ("A", "B"):
        for method in ("pearson", "spearman"):
            rows.append(
                {
                    "scope": scope,
                    "coefficient": coefficient,
                    "xray_column": "xrsb",
                    "method": method,
                    "n": len(frame.dropna(subset=[coefficient, "xrsb"])),
                    "correlation": safe_corr(frame[coefficient], frame["xrsb"], method),
                }
            )


def append_plot_sample(
    samples: list[pd.DataFrame], event: str, time: pd.Timestamp, fit: dict, rng: np.random.Generator
) -> None:
    actual = fit["actual"]
    predicted = fit["predicted"]
    count = min(len(actual), max(1, PLOT_SAMPLE_SIZE // 100))
    indices = rng.choice(len(actual), size=count, replace=False)
    samples.append(
        pd.DataFrame(
            {
                "event": event,
                "time": time,
                "actual": actual[indices],
                "predicted": predicted[indices],
            }
        )
    )


def plot_full_event(event_frame: pd.DataFrame, samples: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    event = event_frame["event"].iloc[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(samples["actual"], samples["predicted"], s=5, alpha=0.2)
    low = min(samples["actual"].min(), samples["predicted"].min())
    high = max(samples["actual"].max(), samples["predicted"].max())
    ax.plot([low, high], [low, high], "k--", linewidth=1)
    ax.set(xlabel="Real ROTI", ylabel="Predicted ROTI", title=f"{event}: predicted vs real ROTI")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "predicted_vs_real_roti.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, coefficient, color in zip(axes, ("A", "B"), ("#0057B8", "#D62728")):
        ax.scatter(event_frame["xrsb"] * 1e6, event_frame[coefficient], s=18, alpha=0.75, color=color)
        pearson = safe_corr(event_frame["xrsb"], event_frame[coefficient], "pearson")
        spearman = safe_corr(event_frame["xrsb"], event_frame[coefficient], "spearman")
        ax.set(
            xlabel="GOES XRSB, 10^-6 W/m^2",
            ylabel=coefficient,
            title=f"{coefficient} vs XRSB\nPearson={pearson:.3f}, Spearman={spearman:.3f}",
        )
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ab_vs_xray.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(event_frame["time"], event_frame["xrsb"] * 1e6, color="black")
    axes[0].set_ylabel("XRSB, 10^-6 W/m^2")
    axes[1].plot(event_frame["time"], event_frame["A"], color="#0057B8")
    axes[1].set_ylabel("A")
    axes[2].plot(event_frame["time"], event_frame["B"], color="#D62728")
    axes[2].set_ylabel("B")
    axes[2].set_xlabel("UTC")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"{event}: X-ray and fitted coefficients")
    fig.tight_layout()
    fig.savefig(output_dir / "xray_ab_time_series.png", dpi=180)
    plt.close(fig)


def build_full_flare_statistics(
    events: list[dict], output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slice_rows = []
    correlation_rows: list[dict] = []
    errors = []
    rng = np.random.default_rng(42)

    for event_index, event_info in enumerate(events, 1):
        event = event_info["event"]
        print(f"[full flare {event_index}/{len(events)}] {event}")
        try:
            xray = load_xray(event_info["event_dir"])
            event_rows = []
            samples: list[pd.DataFrame] = []
            with h5py.File(event_info["map_path"], "r") as handle:
                for time_key in sorted(handle["data"].keys()):
                    time = parse_time(time_key)
                    fit = fit_slice(handle["data"][time_key], time)
                    if fit is None:
                        errors.append(
                            {"event": event, "stage": "full_flare_slice", "time": time, "error": "fit unavailable"}
                        )
                        continue
                    event_rows.append(
                        {
                            "event": event,
                            "map_path": str(event_info["map_path"]),
                            "time": time,
                            **{key: fit[key] for key in SLICE_COLUMNS if key in fit},
                        }
                    )
                    append_plot_sample(samples, event, time, fit, rng)
        except (KeyError, OSError, ValueError, pd.errors.ParserError) as exc:
            errors.append({"event": event, "stage": "full_flare", "error": str(exc)})
            continue

        event_frame = pd.DataFrame(event_rows)
        if event_frame.empty:
            errors.append({"event": event, "stage": "full_flare", "error": "no usable ROTI slices"})
            continue
        event_frame["xrsb"] = interpolate_xray(xray, event_frame["time"].tolist())
        event_frame = event_frame[SLICE_COLUMNS]
        slice_rows.extend(event_frame.to_dict("records"))
        add_correlation_rows(correlation_rows, event_frame, event)
        plot_full_event(event_frame, pd.concat(samples, ignore_index=True), output_dir / event)

    return pd.DataFrame(slice_rows, columns=SLICE_COLUMNS), pd.DataFrame(correlation_rows), pd.DataFrame(errors)


def build_peak_statistics(
    events: list[dict], output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    errors = []
    for event_index, event_info in enumerate(events, 1):
        event = event_info["event"]
        print(f"[peak {event_index}/{len(events)}] {event}")
        peak_time = parse_peak_time(event)
        if peak_time is None or pd.isna(peak_time):
            errors.append({"event": event, "stage": "peak", "error": "cannot parse peak time from event name"})
            continue
        try:
            xray = load_xray(event_info["event_dir"])
            with h5py.File(event_info["map_path"], "r") as handle:
                keys = sorted(handle["data"].keys())
                map_time = min((parse_time(key) for key in keys), key=lambda value: abs(value - peak_time))
                map_key = min(keys, key=lambda key: abs(parse_time(key) - peak_time))
                fit = fit_slice(handle["data"][map_key], map_time)
        except (KeyError, OSError, ValueError, pd.errors.ParserError) as exc:
            errors.append({"event": event, "stage": "peak", "error": str(exc)})
            continue
        if fit is None:
            errors.append({"event": event, "stage": "peak", "error": "fit unavailable at nearest peak slice"})
            continue
        rows.append(
            {
                "event": event,
                "map_path": str(event_info["map_path"]),
                "peak_time": peak_time,
                "map_time": map_time,
                "map_peak_delta_seconds": abs((map_time - peak_time).total_seconds()),
                "xrsb": interpolate_xray(xray, [peak_time])[0],
                "xrsb_at_map_time": interpolate_xray(xray, [map_time])[0],
                **{key: fit[key] for key in SLICE_COLUMNS if key in fit},
            }
        )

    peak_frame = pd.DataFrame(rows)
    correlation_rows: list[dict] = []
    if not peak_frame.empty:
        add_correlation_rows(correlation_rows, peak_frame, "all_peak_flares")
        plot_peak_correlations(peak_frame, output_dir)
    return peak_frame, pd.DataFrame(correlation_rows), pd.DataFrame(errors)


def build_event_summary(full_stats: pd.DataFrame) -> pd.DataFrame:
    if full_stats.empty:
        return pd.DataFrame()
    return (
        full_stats.groupby("event", as_index=False)
        .agg(
            slices=("time", "count"),
            total_day_points=("n_points", "sum"),
            actual_roti_mean=("actual_mean", "mean"),
            predicted_roti_mean=("predicted_mean", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_r2=("r2", "mean"),
            mean_actual_predicted_r=("pearson_actual_predicted", "mean"),
            A_mean=("A", "mean"),
            B_mean=("B", "mean"),
            xrsb_mean=("xrsb", "mean"),
            xrsb_max=("xrsb", "max"),
        )
        .sort_values("event")
    )


def plot_peak_correlations(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, coefficient, color in zip(axes, ("A", "B"), ("#0057B8", "#D62728")):
        ax.scatter(frame["xrsb"] * 1e6, frame[coefficient], s=55, alpha=0.8, color=color)
        for _, row in frame.iterrows():
            ax.annotate(row["event"], (row["xrsb"] * 1e6, row[coefficient]), fontsize=7)
        pearson = safe_corr(frame["xrsb"], frame[coefficient], "pearson")
        spearman = safe_corr(frame["xrsb"], frame[coefficient], "spearman")
        ax.set(
            xlabel="GOES XRSB at flare peak, 10^-6 W/m^2",
            ylabel=coefficient,
            title=f"Peak {coefficient} vs XRSB\nPearson={pearson:.3f}, Spearman={spearman:.3f}",
        )
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "peak_ab_vs_xray.png", dpi=180)
    plt.close(fig)


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Saved: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROTI fit statistics derived from analysis/roti_fit.ipynb.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--event", help="Process only an event whose directory name contains this value.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    all_events = load_events(results_dir)
    events = select_usable_events(results_dir, all_events)
    if args.event:
        events = [event for event in events if args.event in event["event"]]
    if not events:
        raise SystemExit("No events with both maps/map_roti.h5 and goes_xray/goes_xray.csv were found.")

    print(f"Events in results: {len(all_events)}")
    print(f"Events with GOES X-ray and ROTI map: {len(events)}")
    output_dir = args.output_dir.resolve()
    full_dir = output_dir / "full_flare"
    peak_dir = output_dir / "peak_flares"
    full_stats, full_correlations, full_errors = build_full_flare_statistics(events, full_dir)
    peak_stats, peak_correlations, peak_errors = build_peak_statistics(events, peak_dir)
    event_summary = build_event_summary(full_stats)
    errors = pd.concat([full_errors, peak_errors], ignore_index=True)

    save_frame(full_stats, full_dir / "roti_fit_slice_statistics.csv")
    save_frame(event_summary, full_dir / "roti_fit_event_summary.csv")
    save_frame(full_correlations, full_dir / "ab_xray_correlations.csv")
    save_frame(peak_stats, peak_dir / "peak_ab_statistics.csv")
    save_frame(peak_correlations, peak_dir / "peak_ab_xray_correlations.csv")
    save_frame(errors, output_dir / "roti_fit_errors.csv")

    print(f"Events with ROTI maps: {len(events)}")
    print(f"Full-flare slices: {len(full_stats)}")
    print(f"Peak-flare rows: {len(peak_stats)}")
    print(f"Errors/skips: {len(errors)}")
    if len(peak_stats) < 2:
        print("Peak correlation is undefined until at least two events have ROTI maps.")


if __name__ == "__main__":
    main()
