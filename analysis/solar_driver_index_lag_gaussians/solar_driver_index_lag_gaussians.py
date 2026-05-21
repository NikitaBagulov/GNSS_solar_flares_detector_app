from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
LAGS_ANALYSIS_DIR = REPO_ROOT / "analysis" / "solar_driver_index_lags"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

sys.path.insert(0, str(LAGS_ANALYSIS_DIR))
from solar_driver_index_lags import (  # noqa: E402
    DRIVER_COLORS,
    DRIVER_LABELS,
    INDEX_COLUMNS,
    PRODUCTS,
    SOLAR_DRIVERS,
    build_lag_rows,
    load_events,
)


def gaussian_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std <= 0 or pd.isna(std):
        return np.full_like(x, np.nan, dtype=float)
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * math.sqrt(2.0 * math.pi))


def filter_lags(lags: pd.DataFrame, max_abs_lag_minutes: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lags.empty or max_abs_lag_minutes is None:
        return lags.copy(), pd.DataFrame(columns=lags.columns)

    valid = lags["lag_minutes"].abs() <= max_abs_lag_minutes
    return lags[valid].copy(), lags[~valid].copy()


def build_gaussian_summary(lags: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "driver",
        "driver_label",
        "product",
        "index",
        "n",
        "lag_minutes_mean",
        "lag_minutes_std",
        "lag_minutes_median",
        "lag_minutes_min",
        "lag_minutes_max",
        "lag_minutes_q25",
        "lag_minutes_q75",
    ]
    if lags.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (driver, product, index_column), subset in lags.groupby(["driver", "product", "index"]):
        values = subset["lag_minutes"].dropna()
        if values.empty:
            continue
        rows.append(
            {
                "driver": driver,
                "driver_label": DRIVER_LABELS.get(driver, driver),
                "product": product,
                "index": index_column,
                "n": int(values.count()),
                "lag_minutes_mean": float(values.mean()),
                "lag_minutes_std": float(values.std(ddof=1)) if len(values) >= 2 else np.nan,
                "lag_minutes_median": float(values.median()),
                "lag_minutes_min": float(values.min()),
                "lag_minutes_max": float(values.max()),
                "lag_minutes_q25": float(values.quantile(0.25)),
                "lag_minutes_q75": float(values.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["driver", "index", "product"])


def plot_gaussian_distribution(
    ax,
    values: pd.Series,
    driver: str,
    title: str,
    bins: int,
) -> None:
    values = values.dropna()
    if values.empty:
        ax.axis("off")
        return

    mean = values.mean()
    std = values.std(ddof=1) if len(values) >= 2 else np.nan
    median = values.median()
    color = DRIVER_COLORS.get(driver, "#4c78a8")

    ax.hist(values, bins=min(bins, max(3, len(values))), density=True, alpha=0.42, color=color, edgecolor="white")
    if len(values) >= 2 and pd.notna(std) and std > 0:
        x_min = min(values.min(), mean - 3.0 * std)
        x_max = max(values.max(), mean + 3.0 * std)
        x = np.linspace(float(x_min), float(x_max), 300)
        ax.plot(x, gaussian_pdf(x, float(mean), float(std)), color=color, linewidth=2.4, label=f"Gaussian: mu={mean:.1f}, sigma={std:.1f}")

    ax.axvline(0, color="black", linewidth=0.9, alpha=0.55)
    ax.axvline(mean, color=color, linestyle="-", linewidth=1.8, alpha=0.95, label=f"mean {mean:.1f} min")
    ax.axvline(median, color="black", linestyle="--", linewidth=1.4, alpha=0.8, label=f"median {median:.1f} min")
    ax.set_title(title)
    ax.set_xlabel("Index peak time - solar driver peak time, minutes")
    ax.set_ylabel("Probability density")
    ax.text(0.03, 0.95, f"n={len(values)}", transform=ax.transAxes, va="top")
    ax.legend(fontsize=8)


def plot_by_driver_product(lags: pd.DataFrame, output_dir: Path, bins: int) -> None:
    if lags.empty:
        return

    for driver in SOLAR_DRIVERS:
        driver_df = lags[lags["driver"] == driver]
        if driver_df.empty:
            continue
        for index_column in INDEX_COLUMNS:
            index_df = driver_df[driver_df["index"] == index_column]
            if index_df.empty:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), squeeze=False)
            axes_flat = axes.ravel()
            for ax, product in zip(axes_flat, PRODUCTS):
                values = index_df[index_df["product"] == product]["lag_minutes"]
                plot_gaussian_distribution(
                    ax=ax,
                    values=values,
                    driver=driver,
                    title=f"{DRIVER_LABELS.get(driver, driver)} -> {product}",
                    bins=bins,
                )

            fig.suptitle(f"Lag distribution: {DRIVER_LABELS.get(driver, driver)} to {index_column} maximum", fontsize=15)
            fig.tight_layout()
            fig.savefig(output_dir / f"gaussian_lag_{driver}_{index_column}.png", dpi=160)
            plt.close(fig)


def plot_driver_comparison(lags: pd.DataFrame, output_dir: Path, bins: int) -> None:
    if lags.empty:
        return

    for product in PRODUCTS:
        product_df = lags[lags["product"] == product]
        if product_df.empty:
            continue
        for index_column in INDEX_COLUMNS:
            index_df = product_df[product_df["index"] == index_column]
            if index_df.empty:
                continue

            fig, axes = plt.subplots(len(SOLAR_DRIVERS), 1, figsize=(11, 9.5), squeeze=False)
            for ax, driver in zip(axes.ravel(), SOLAR_DRIVERS):
                values = index_df[index_df["driver"] == driver]["lag_minutes"]
                plot_gaussian_distribution(
                    ax=ax,
                    values=values,
                    driver=driver,
                    title=DRIVER_LABELS.get(driver, driver),
                    bins=bins,
                )

            fig.suptitle(f"{product}: lag distributions to {index_column} maximum", fontsize=15)
            fig.tight_layout()
            fig.savefig(output_dir / f"gaussian_lag_driver_comparison_{product}_{index_column}.png", dpi=160)
            plt.close(fig)


def plot_summary_means(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return

    for index_column in INDEX_COLUMNS:
        index_df = summary[summary["index"] == index_column]
        if index_df.empty:
            continue

        fig, axes = plt.subplots(1, len(SOLAR_DRIVERS), figsize=(15, 4.8), squeeze=False)
        for ax, driver in zip(axes.ravel(), SOLAR_DRIVERS):
            driver_df = index_df[index_df["driver"] == driver].set_index("product").reindex(PRODUCTS)
            driver_df = driver_df.dropna(subset=["lag_minutes_mean"])
            if driver_df.empty:
                ax.axis("off")
                continue

            means = driver_df["lag_minutes_mean"]
            errors = driver_df["lag_minutes_std"].fillna(0.0)
            y = np.arange(len(driver_df))
            ax.barh(y, means, xerr=errors, color=DRIVER_COLORS.get(driver, "#4c78a8"), alpha=0.82)
            ax.axvline(0, color="black", linewidth=0.9)
            ax.set_yticks(y, driver_df.index)
            ax.set_title(DRIVER_LABELS.get(driver, driver))
            ax.set_xlabel("Gaussian mean lag, minutes")
            for row_idx, (_, row) in enumerate(driver_df.iterrows()):
                ax.text(
                    row["lag_minutes_mean"],
                    row_idx,
                    f" n={int(row['n'])}",
                    va="center",
                    ha="left" if row["lag_minutes_mean"] >= 0 else "right",
                    fontsize=8.5,
                )

        fig.suptitle(f"Gaussian lag mean +/- sigma for {index_column}", fontsize=15)
        fig.tight_layout()
        fig.savefig(output_dir / f"gaussian_lag_mean_summary_{index_column}.png", dpi=160)
        plt.close(fig)


def save_outputs(
    lags: pd.DataFrame,
    excluded_lags: pd.DataFrame,
    errors: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    make_plots: bool,
    bins: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    lags_path = output_dir / "solar_driver_index_lag_gaussians.csv"
    excluded_path = output_dir / "solar_driver_index_lag_gaussian_excluded.csv"
    errors_path = output_dir / "solar_driver_index_lag_gaussian_errors.csv"
    summary_path = output_dir / "solar_driver_index_lag_gaussian_summary.csv"

    lags.to_csv(lags_path, index=False)
    excluded_lags.to_csv(excluded_path, index=False)
    errors.to_csv(errors_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {lags_path}")
    print(f"Saved: {excluded_path}")
    print(f"Saved: {errors_path}")
    print(f"Saved: {summary_path}")

    if make_plots:
        plt.style.use("seaborn-v0_8-whitegrid")
        plot_by_driver_product(lags, output_dir, bins=bins)
        plot_driver_comparison(lags, output_dir, bins=bins)
        plot_summary_means(summary, output_dir)
        print(f"Saved plots to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Gaussian lag distributions between solar drivers and GNSS index peaks.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--xray-column", default="xrsb")
    parser.add_argument("--euv-column", default="flux_01_50")
    parser.add_argument("--window-padding-minutes", type=float, default=10.0)
    parser.add_argument("--max-abs-lag-minutes", type=float, default=None)
    parser.add_argument("--bins", type=int, default=14)
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
    lags, excluded_lags = filter_lags(lags, args.max_abs_lag_minutes)
    summary = build_gaussian_summary(lags)
    save_outputs(
        lags=lags,
        excluded_lags=excluded_lags,
        errors=errors,
        summary=summary,
        output_dir=output_dir,
        make_plots=not args.no_plots,
        bins=args.bins,
    )

    print(f"Lag rows used: {len(lags)}")
    print(f"Lag rows excluded: {len(excluded_lags)}")
    print(f"Errors/skips: {len(errors)}")
    if not summary.empty:
        print("\nGaussian lag summary:")
        print(summary[["driver", "product", "index", "n", "lag_minutes_mean", "lag_minutes_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
