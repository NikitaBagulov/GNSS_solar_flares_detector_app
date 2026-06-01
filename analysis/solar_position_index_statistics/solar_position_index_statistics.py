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
DEFAULT_FLARES_CSV = REPO_ROOT / "data" / "all_flares.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_SOLAR_RADIUS_ARCSEC = 960.0
DEFAULT_WINDOW_PADDING_MINUTES = 10.0

sys.path.insert(0, str(XRAY_STATS_DIR))
from xray_index_peak_statistics import (  # noqa: E402
    FLARE_CLASS_MARKERS,
    INDEX_COLUMNS,
    PLOTTED_FLARE_CLASSES,
    PRODUCTS,
    build_statistics,
    flare_class_letter,
    load_events,
    normalize_time_column,
)


def event_file_path(results_dir: Path, event: dict, *parts: str) -> Path:
    return results_dir / event["path"] / Path(*parts)


def load_flares_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(path)
    for column in ("start_time", "peak_time", "end_time"):
        if column in catalog.columns:
            catalog[column] = pd.to_datetime(catalog[column], utc=True, errors="coerce")
    for column in ("hpc_x", "hpc_y", "class_value", "duration_min", "peak_flux"):
        if column in catalog.columns:
            catalog[column] = pd.to_numeric(catalog[column], errors="coerce")
    if "date" in catalog.columns:
        catalog["date"] = pd.to_datetime(catalog["date"], errors="coerce").dt.date.astype("string")
    if "class" in catalog.columns:
        catalog["class"] = catalog["class"].astype("string").str.upper()
    return catalog


def parse_readable_event_name(name: str) -> tuple[str | None, str | None]:
    match = re.match(r"(\d{4}-\d{2}-\d{2})_([ABCMX]\d+(?:\.\d+)?)", name.upper())
    if not match:
        return None, None
    return match.group(1), match.group(2)


def match_flare_row(event: dict, catalog: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    event_name = str(event.get("name", ""))
    if "flare_key" in catalog.columns:
        exact = catalog[catalog["flare_key"].astype(str) == event_name]
        if len(exact) == 1:
            return exact.iloc[0], None
        if len(exact) > 1:
            return exact.iloc[0], f"multiple catalog rows for flare_key {event_name!r}; used first"

    event_date, event_class = parse_readable_event_name(event_name)
    if event_date and event_class and {"date", "class"}.issubset(catalog.columns):
        matched = catalog[(catalog["date"] == event_date) & (catalog["class"] == event_class)]
        if len(matched) == 1:
            return matched.iloc[0], None
        if len(matched) > 1:
            return matched.iloc[0], f"multiple catalog rows for {event_date} {event_class}; used first"

    return None, f"no catalog row for event {event_name!r}"


def load_soho_sem(results_dir: Path, event: dict, euv_column: str) -> pd.DataFrame:
    path = event_file_path(results_dir, event, "soho_sem", "soho_sem.csv")
    df = normalize_time_column(pd.read_csv(path), preferred="time")
    if euv_column not in df.columns:
        raise ValueError(f"SOHO SEM CSV has no column {euv_column!r}")
    df[euv_column] = pd.to_numeric(df[euv_column], errors="coerce")
    return df.dropna(subset=[euv_column])


def clipped_peak(
    frame: pd.DataFrame,
    value_column: str,
    start_time: pd.Timestamp | None,
    end_time: pd.Timestamp | None,
    padding: pd.Timedelta,
) -> tuple[pd.Timestamp | pd.NaT, float]:
    data = frame
    if start_time is not None and end_time is not None and pd.notna(start_time) and pd.notna(end_time):
        data = frame[(frame["time"] >= start_time - padding) & (frame["time"] <= end_time + padding)]
    if data.empty:
        data = frame
    values = pd.to_numeric(data[value_column], errors="coerce")
    if values.dropna().empty:
        return pd.NaT, np.nan
    row = data.loc[values.idxmax()]
    return row["time"], float(row[value_column])


def build_solar_parameters(
    results_dir: Path,
    events: list[dict],
    catalog: pd.DataFrame,
    euv_column: str,
    window_padding: pd.Timedelta,
    solar_radius_arcsec: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    errors = []

    for event in events:
        flare_row, warning = match_flare_row(event, catalog)
        if warning:
            errors.append({"event": event.get("name"), "stage": "catalog", "error": warning})
        if flare_row is None:
            continue

        hpc_x = pd.to_numeric(flare_row.get("hpc_x"), errors="coerce")
        hpc_y = pd.to_numeric(flare_row.get("hpc_y"), errors="coerce")
        if pd.isna(hpc_x) or pd.isna(hpc_y):
            errors.append({"event": event.get("name"), "stage": "catalog", "error": "missing hpc_x/hpc_y"})
            continue

        radial_arcsec = math.hypot(float(hpc_x), float(hpc_y))
        radial_fraction = radial_arcsec / solar_radius_arcsec
        mu = math.sqrt(max(0.0, 1.0 - radial_fraction**2)) if radial_fraction <= 1.0 else np.nan
        center_angle_deg = math.degrees(math.asin(min(1.0, radial_fraction)))

        start_time = flare_row.get("start_time")
        end_time = flare_row.get("end_time")
        euv_peak_time = pd.NaT
        euv_peak_flux = np.nan
        if event.get("sources", {}).get("soho_sem"):
            try:
                euv = load_soho_sem(results_dir, event, euv_column)
                euv_peak_time, euv_peak_flux = clipped_peak(euv, euv_column, start_time, end_time, window_padding)
            except (ValueError, OSError, pd.errors.ParserError) as exc:
                errors.append({"event": event.get("name"), "stage": "euv", "error": str(exc)})

        rows.append(
            {
                "event": event["name"],
                "event_path": event["path"],
                "catalog_flare_key": flare_row.get("flare_key"),
                "flare_class": flare_row.get("class", event.get("class")),
                "class_value": flare_row.get("class_value"),
                "start_time": start_time,
                "peak_time": flare_row.get("peak_time"),
                "end_time": end_time,
                "duration_min": flare_row.get("duration_min"),
                "hpc_x": float(hpc_x),
                "hpc_y": float(hpc_y),
                "radial_distance_arcsec": radial_arcsec,
                "radial_fraction": radial_fraction,
                "mu": mu,
                "center_angle_deg": center_angle_deg,
                "euv_column": euv_column,
                "euv_peak_time": euv_peak_time,
                "euv_peak_flux": euv_peak_flux,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(errors)


def build_position_statistics(
    index_stats: pd.DataFrame,
    solar_params: pd.DataFrame,
) -> pd.DataFrame:
    if index_stats.empty or solar_params.empty:
        return pd.DataFrame()
    merged = index_stats.merge(solar_params, on=["event", "event_path"], how="inner", suffixes=("", "_catalog"))
    return merged


def build_correlations(stats: pd.DataFrame) -> pd.DataFrame:
    columns = ["variable", "product", "index", "lag_seconds", "lag_minutes", "n", "spearman_r"]
    if stats.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    variables = ["radial_fraction", "mu", "center_angle_deg", "hpc_x", "hpc_y"]
    for variable in variables:
        if variable not in stats.columns:
            continue
        for (product, lag_seconds, lag_minutes), product_df in stats.groupby(["product", "lag_seconds", "lag_minutes"]):
            for index_column in INDEX_COLUMNS:
                subset = product_df.dropna(subset=[variable, index_column])
                corr = np.nan
                if len(subset) >= 2:
                    corr = subset[[variable, index_column]].corr(method="spearman").iloc[0, 1]
                rows.append(
                    {
                        "variable": variable,
                        "product": product,
                        "index": index_column,
                        "lag_seconds": lag_seconds,
                        "lag_minutes": lag_minutes,
                        "n": len(subset),
                        "spearman_r": corr,
                    }
                )
    return pd.DataFrame(rows, columns=columns).sort_values(["variable", "index", "product", "lag_seconds"])


def plot_indices_vs_position(stats: pd.DataFrame, output_dir: Path) -> None:
    if stats.empty:
        return
    data = stats.copy()
    if "lag_seconds" in data.columns:
        data = data[data["lag_seconds"] == 0]
    data["flare_class_letter"] = data["flare_class"].map(flare_class_letter)
    data = data[data["flare_class_letter"].isin(PLOTTED_FLARE_CLASSES)]
    if data.empty:
        return

    for index_column in INDEX_COLUMNS:
        plot_data = data.dropna(subset=["radial_fraction", index_column])
        if plot_data.empty:
            continue
        products = [product for product in PRODUCTS if product in set(plot_data["product"])]
        cols = 2
        rows_count = math.ceil(len(products) / cols)
        fig, axes = plt.subplots(rows_count, cols, figsize=(13, 4.8 * rows_count), squeeze=False)
        axes_flat = axes.ravel()

        for ax, product in zip(axes_flat, products):
            subset = plot_data[plot_data["product"] == product]
            for flare_class in PLOTTED_FLARE_CLASSES:
                class_subset = subset[subset["flare_class_letter"] == flare_class]
                if class_subset.empty:
                    continue
                ax.scatter(
                    class_subset["radial_fraction"],
                    class_subset[index_column],
                    s=64,
                    alpha=0.82,
                    marker=FLARE_CLASS_MARKERS[flare_class],
                    label=f"{flare_class}-class",
                )
            ax.set_title(product)
            ax.set_xlabel("Distance from disk center, R_sun")
            ax.set_ylabel(index_column)
            ax.legend(title="Flare class")

        for ax in axes_flat[len(products):]:
            ax.axis("off")

        fig.suptitle(f"{index_column} vs flare position on solar disk", fontsize=15)
        fig.tight_layout()
        fig.savefig(output_dir / f"{index_column}_vs_solar_disk_position.png", dpi=160)
        plt.close(fig)


def plot_solar_drivers_vs_position(solar_params: pd.DataFrame, index_stats: pd.DataFrame, output_dir: Path) -> None:
    if solar_params.empty:
        return

    xray_by_event = (
        index_stats[index_stats["lag_seconds"] == 0][["event", "xray_at_flare_peak"]]
        .dropna()
        .drop_duplicates("event")
        if not index_stats.empty and {"lag_seconds", "xray_at_flare_peak"}.issubset(index_stats.columns)
        else pd.DataFrame(columns=["event", "xray_at_flare_peak"])
    )
    data = solar_params.merge(xray_by_event, on="event", how="left")
    data["flare_class_letter"] = data["flare_class"].map(flare_class_letter)
    data = data[data["flare_class_letter"].isin(PLOTTED_FLARE_CLASSES)]

    variables = [
        ("xray_at_flare_peak", "GOES X-ray at flare peak, W/m^2", "xray_vs_solar_disk_position.png", True),
        ("euv_peak_flux", "SOHO SEM EUV peak flux", "euv_vs_solar_disk_position.png", False),
    ]
    for value_column, ylabel, filename, log_y in variables:
        plot_data = data.dropna(subset=["radial_fraction", value_column])
        if plot_data.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for flare_class in PLOTTED_FLARE_CLASSES:
            class_subset = plot_data[plot_data["flare_class_letter"] == flare_class]
            if class_subset.empty:
                continue
            ax.scatter(
                class_subset["radial_fraction"],
                class_subset[value_column],
                s=72,
                alpha=0.82,
                marker=FLARE_CLASS_MARKERS[flare_class],
                label=f"{flare_class}-class",
            )
        ax.set_xlabel("Distance from disk center, R_sun")
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
        ax.set_title(f"{ylabel} vs flare position on solar disk")
        ax.legend(title="Flare class")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def plot_solar_disk_maps(solar_params: pd.DataFrame, index_stats: pd.DataFrame, output_dir: Path) -> None:
    if solar_params.empty:
        return

    xray_by_event = (
        index_stats[index_stats["lag_seconds"] == 0][["event", "xray_at_flare_peak"]]
        .dropna()
        .drop_duplicates("event")
        if not index_stats.empty and {"lag_seconds", "xray_at_flare_peak"}.issubset(index_stats.columns)
        else pd.DataFrame(columns=["event", "xray_at_flare_peak"])
    )
    data = solar_params.merge(xray_by_event, on="event", how="left")
    data = data.dropna(subset=["hpc_x", "hpc_y"])
    if data.empty:
        return

    for value_column, title, filename in [
        ("xray_at_flare_peak", "Flare positions colored by GOES X-ray peak", "solar_disk_positions_xray.png"),
        ("euv_peak_flux", "Flare positions colored by SOHO SEM EUV peak", "solar_disk_positions_euv.png"),
    ]:
        plot_data = data.dropna(subset=[value_column])
        if plot_data.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 7))
        disk = plt.Circle((0, 0), DEFAULT_SOLAR_RADIUS_ARCSEC, color="black", fill=False, linewidth=1.2, alpha=0.6)
        ax.add_patch(disk)
        scatter = ax.scatter(plot_data["hpc_x"], plot_data["hpc_y"], c=plot_data[value_column], s=58, cmap="viridis", alpha=0.88)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("HPC X, arcsec")
        ax.set_ylabel("HPC Y, arcsec")
        ax.set_title(title)
        fig.colorbar(scatter, ax=ax, label=value_column)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def save_outputs(
    stats: pd.DataFrame,
    solar_params: pd.DataFrame,
    errors: pd.DataFrame,
    correlations: pd.DataFrame,
    output_dir: Path,
    make_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "solar_position_index_statistics.csv"
    solar_params_path = output_dir / "solar_position_solar_parameters.csv"
    errors_path = output_dir / "solar_position_errors.csv"
    correlations_path = output_dir / "solar_position_index_correlations.csv"

    stats.to_csv(stats_path, index=False)
    solar_params.to_csv(solar_params_path, index=False)
    errors.to_csv(errors_path, index=False)
    correlations.to_csv(correlations_path, index=False)

    print(f"Saved: {stats_path}")
    print(f"Saved: {solar_params_path}")
    print(f"Saved: {errors_path}")
    print(f"Saved: {correlations_path}")

    if make_plots:
        plt.style.use("seaborn-v0_8-whitegrid")
        plot_indices_vs_position(stats, output_dir)
        plot_solar_drivers_vs_position(solar_params, stats, output_dir)
        plot_solar_disk_maps(solar_params, stats, output_dir)
        print(f"Saved plots to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze solar radiation and GNSS indices vs flare position on solar disk.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--flares-csv", type=Path, default=DEFAULT_FLARES_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xray-column", default="xrsb")
    parser.add_argument("--euv-column", default="flux_01_50")
    parser.add_argument("--solar-radius-arcsec", type=float, default=DEFAULT_SOLAR_RADIUS_ARCSEC)
    parser.add_argument("--window-padding-minutes", type=float, default=DEFAULT_WINDOW_PADDING_MINUTES)
    parser.add_argument("--peak-time-source", choices=["event_name", "goes_max"], default="event_name")
    parser.add_argument("--max-time-delta-seconds", type=float, default=90.0)
    parser.add_argument("--max-index-lag-minutes", type=float, default=10.0)
    parser.add_argument("--lag-step-seconds", type=float, default=60.0)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    flares_csv = args.flares_csv.resolve()
    output_dir = args.output_dir.resolve()

    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")
    if not flares_csv.exists():
        raise SystemExit(f"Flares catalog does not exist: {flares_csv}")

    events = load_events(results_dir)
    catalog = load_flares_catalog(flares_csv)
    index_stats, index_errors = build_statistics(
        results_dir=results_dir,
        events=events,
        xray_column=args.xray_column,
        peak_time_source=args.peak_time_source,
        max_time_delta=pd.Timedelta(seconds=args.max_time_delta_seconds),
        max_index_lag=pd.Timedelta(minutes=args.max_index_lag_minutes),
        lag_step=pd.Timedelta(seconds=args.lag_step_seconds),
    )
    solar_params, solar_errors = build_solar_parameters(
        results_dir=results_dir,
        events=events,
        catalog=catalog,
        euv_column=args.euv_column,
        window_padding=pd.Timedelta(minutes=args.window_padding_minutes),
        solar_radius_arcsec=args.solar_radius_arcsec,
    )
    stats = build_position_statistics(index_stats, solar_params)
    correlations = build_correlations(stats)
    errors = pd.concat([index_errors, solar_errors], ignore_index=True)

    save_outputs(
        stats=stats,
        solar_params=solar_params,
        errors=errors,
        correlations=correlations,
        output_dir=output_dir,
        make_plots=not args.no_plots,
    )

    print(f"Events in results: {len(events)}")
    print(f"Events with solar position: {solar_params['event'].nunique() if not solar_params.empty else 0}")
    print(f"Statistics rows: {len(stats)}")
    print(f"Errors/skips: {len(errors)}")


if __name__ == "__main__":
    main()
