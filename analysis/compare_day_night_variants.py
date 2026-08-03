"""Compare Day/Night index definitions without overwriting production indices.

Example:
    python3 -m analysis.compare_day_night_variants \\
      --results-dir results --events 2025-11-11_X5.1 \\
      --output-dir analysis/article_outputs/day_night_pilot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from index_functions.day_night_index import DAY_NIGHT_VARIANTS, compute_day_night_components

PRODUCTS = ("roti", "dtec_2_10", "dtec_10_20", "dtec_20_60")


def find_event(results_dir: Path, name: str) -> Path:
    matches = [path for path in results_dir.glob(f"*/{name}") if path.is_dir()]
    if not matches and (results_dir / name).is_dir():
        matches = [results_dir / name]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one directory for {name}, found {len(matches)}")
    return matches[0]


def read_map(path: Path):
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            return
        for time_key, dataset in handle["data"].items():
            timestamp = pd.Timestamp(time_key).to_pydatetime()
            points = dataset[:]
            if points.dtype.names and {"lat", "lon", "vals"}.issubset(points.dtype.names):
                values = np.column_stack((points["lat"], points["lon"], points["vals"]))
            else:
                values = np.asarray(points)
            yield timestamp, values


def compare_event(event_dir: Path, products, epsilons, exclusions, output_dir: Path):
    rows = []
    for product in products:
        map_path = event_dir / "maps" / f"map_{product}.h5"
        if not map_path.is_file():
            print(f"WARNING missing {map_path}")
            continue
        for timestamp, points in read_map(map_path):
            for variant in DAY_NIGHT_VARIANTS:
                settings = [(np.nan, 0.0)]
                if variant == "distance_weight_cos":
                    settings = [(epsilon, exclusion) for epsilon in epsilons for exclusion in exclusions]
                for epsilon, exclusion in settings:
                    components = compute_day_night_components(
                        points, timestamp, variant=variant,
                        eps_abs=1e-6 if np.isnan(epsilon) else epsilon,
                        exclude_terminator_deg=exclusion,
                    )
                    if components is None:
                        continue
                    rows.append({
                        "event": event_dir.name,
                        "product": product,
                        "time": timestamp,
                        "variant": variant,
                        "epsilon": epsilon,
                        "exclude_terminator_deg": exclusion,
                        **components,
                    })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return False
    event_output = output_dir / event_dir.name
    event_output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(event_output / "day_night_variants_timeseries.csv", index=False)
    group_columns = ["event", "product", "variant", "epsilon", "exclude_terminator_deg"]
    summary = frame.groupby(group_columns, dropna=False)["index"].agg(
        count="count", median="median", q25=lambda x: x.quantile(.25),
        q75=lambda x: x.quantile(.75), minimum="min", maximum="max",
    ).reset_index()
    summary.to_csv(event_output / "day_night_variants_summary.csv", index=False)

    for product, data in frame.groupby("product"):
        fig, ax = plt.subplots(figsize=(11, 6))
        for keys, series in data.groupby(["variant", "epsilon", "exclude_terminator_deg"], dropna=False):
            variant, epsilon, exclusion = keys
            label = variant
            if variant == "distance_weight_cos":
                label += f" (eps={epsilon:g}, exclude={exclusion:g} deg)"
            series = series.sort_values("time")
            ax.plot(series["time"], series["index"], label=label, linewidth=1.3)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(f"{event_dir.name}: {product} Day/Night variants")
        ax.set_ylabel("Normalized day-night contrast")
        ax.set_xlabel("UTC")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(event_output / f"day_night_variants_{product}.png", dpi=160)
        plt.close(fig)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--products", nargs="+", default=list(PRODUCTS), choices=PRODUCTS)
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1e-2, 1e-3, 1e-4, 1e-6])
    parser.add_argument("--exclude-terminator-deg", nargs="+", type=float, default=[0.0, 5.0])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    completed = 0
    for event in args.events:
        event_dir = find_event(args.results_dir, event)
        completed += compare_event(
            event_dir, args.products, args.epsilons,
            args.exclude_terminator_deg, args.output_dir,
        )
    print(f"Completed {completed}/{len(args.events)} events; existing indices were not modified.")
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
