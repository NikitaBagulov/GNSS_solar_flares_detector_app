from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from astropy.time import Time
from sunpy.net import Fido, attrs as a

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flare_utils import build_flare_key
from pipeline.run_config import RunConfig
from pipeline.runner import (
    PipelineConfig,
    run_download_for_date,
    run_index_calculation_for_flare,
    run_preprocessing_for_flares,
)


CLASS_MULTIPLIERS = {
    "X": 10.0,
    "M": 1.0,
    "C": 0.1,
    "B": 0.01,
    "A": 0.001,
}


@dataclass(frozen=True)
class SelectionConfig:
    start_date: date
    end_date: date
    classes: tuple[str, ...]
    per_class: int
    seed: int


def flare_class_to_numeric(flare_class: str) -> float:
    if not isinstance(flare_class, str) or not flare_class.strip():
        return 0.0
    flare_class = flare_class.strip().upper()
    letter = flare_class[0]
    try:
        number = float(flare_class[1:] or "1.0")
    except ValueError:
        number = 1.0
    return CLASS_MULTIPLIERS.get(letter, 0.0) * number


def add_months(value: date, months: int) -> date:
    month = value.month - 1 + months
    year = value.year + month // 12
    month = month % 12 + 1
    day = min(value.day, 28)
    return date(year, month, day)


def iter_date_chunks(start_date: date, end_date: date, chunk_months: int):
    current = start_date
    while current <= end_date:
        next_start = add_months(current, chunk_months)
        chunk_end = min(next_start - timedelta(days=1), end_date)
        yield current, chunk_end
        current = next_start


def normalize_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    if catalog.empty:
        return catalog
    for column in ("start_time", "peak_time", "end_time"):
        catalog[column] = pd.to_datetime(catalog[column], errors="coerce")
    catalog = catalog.dropna(subset=["start_time", "peak_time", "end_time", "class"])
    catalog["date"] = pd.to_datetime(catalog["date"], errors="coerce").dt.date
    if "class_letter" not in catalog.columns:
        catalog["class_letter"] = catalog["class"].astype(str).str.upper().str[0]
    if "class_value" not in catalog.columns:
        catalog["class_value"] = catalog["class"].apply(flare_class_to_numeric)
    if "flare_key" not in catalog.columns:
        catalog["flare_key"] = catalog.apply(
            lambda row: build_flare_key(row["start_time"], row["peak_time"], row["end_time"], row["class"]),
            axis=1,
        )
    return catalog.drop_duplicates(subset=["start_time", "peak_time", "end_time"]).sort_values(
        ["date", "class_letter", "class_value"],
        ascending=[True, True, False],
    )


def fetch_hek_catalog(
    start_date: date,
    end_date: date,
    cache_path: Path,
    min_class: str = "C1.0",
    chunk_months: int = 1,
    retries: int = 3,
    retry_sleep_seconds: float = 20.0,
) -> pd.DataFrame:
    rows = []
    if cache_path.exists():
        existing = pd.read_csv(cache_path)
        if not existing.empty:
            rows.extend(existing.to_dict("records"))

    for chunk_start, chunk_end in iter_date_chunks(start_date, end_date, chunk_months):
        if rows:
            existing_dates = {
                pd.to_datetime(row.get("date"), errors="coerce").date()
                for row in rows
                if pd.notna(pd.to_datetime(row.get("date"), errors="coerce"))
            }
            chunk_dates = {chunk_start + timedelta(days=offset) for offset in range((chunk_end - chunk_start).days + 1)}
            if chunk_dates.issubset(existing_dates):
                continue

        tstart = chunk_start.strftime("%Y/%m/%d 00:00")
        tend = chunk_end.strftime("%Y/%m/%d 23:59")
        print(f"HEK query: {tstart} - {tend}")
        result = None
        for attempt in range(1, retries + 1):
            try:
                result = Fido.search(
                    a.Time(tstart, tend),
                    a.hek.EventType("FL"),
                    a.hek.FL.GOESCls > min_class,
                )
                break
            except Exception as exc:
                print(f"   HEK query failed ({attempt}/{retries}): {exc}")
                if attempt < retries:
                    time.sleep(retry_sleep_seconds)
        if result is None:
            print(f"   Skipping chunk after {retries} failed attempts: {chunk_start}..{chunk_end}")
            continue
        if len(result) == 0:
            continue

        for flare in result["hek"]:
            flare_class = str(flare.get("fl_goescls", "")).strip().upper()
            if not flare_class or flare_class[0] not in {"C", "M"}:
                continue
            try:
                start_time = Time(flare["event_starttime"]).to_datetime()
                peak_time = Time(flare["event_peaktime"]).to_datetime()
                end_time = Time(flare["event_endtime"]).to_datetime()
            except Exception as exc:
                print(f"Skipping malformed HEK row: {exc}")
                continue

            rows.append(
                {
                    "class": flare_class,
                    "class_letter": flare_class[0],
                    "class_value": flare_class_to_numeric(flare_class),
                    "start_time": start_time,
                    "peak_time": peak_time,
                    "end_time": end_time,
                    "duration_min": (end_time - start_time).total_seconds() / 60,
                    "hpc_x": flare.get("hpc_x"),
                    "hpc_y": flare.get("hpc_y"),
                    "peak_flux": flare.get("fl_peakflux"),
                    "date": start_time.date(),
                }
            )

        partial = normalize_catalog(pd.DataFrame(rows))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        partial.to_csv(cache_path, index=False)
        print(f"   Catalog cache updated: {cache_path} ({len(partial)} rows)")

    return normalize_catalog(pd.DataFrame(rows))


def load_or_fetch_catalog(args: argparse.Namespace) -> pd.DataFrame:
    cache_path = args.catalog_cache
    if cache_path.exists() and args.use_cache_only and not args.refresh_catalog:
        print(f"Using cached HEK catalog: {cache_path}")
        catalog = pd.read_csv(cache_path)
    else:
        if cache_path.exists() and args.refresh_catalog:
            cache_path.unlink()
        catalog = fetch_hek_catalog(
            start_date=args.start_date,
            end_date=args.end_date,
            cache_path=cache_path,
            chunk_months=args.chunk_months,
            retries=args.hek_retries,
            retry_sleep_seconds=args.hek_retry_sleep_seconds,
        )
        print(f"Saved HEK catalog cache: {cache_path}")

    return normalize_catalog(catalog)


def uniformly_sample_by_time(
    catalog: pd.DataFrame,
    class_letter: str,
    count: int,
    start_date: date,
    end_date: date,
    seed: int,
) -> pd.DataFrame:
    class_letter = class_letter.upper()
    candidates = catalog[catalog["class_letter"].astype(str).str.upper() == class_letter].copy()
    if candidates.empty:
        return candidates

    candidates["start_time"] = pd.to_datetime(candidates["start_time"])
    candidates = candidates.sort_values("start_time")
    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    bins = pd.date_range(start_ts, end_ts, periods=count + 1)

    chosen_indices = []
    chosen_set = set()
    for left, right in zip(bins[:-1], bins[1:]):
        in_bin = candidates[
            (candidates["start_time"] >= left)
            & (candidates["start_time"] < right)
            & (~candidates.index.isin(chosen_set))
        ]
        if in_bin.empty:
            continue
        picked = rng.choice(in_bin.index.to_numpy())
        chosen_indices.append(picked)
        chosen_set.add(picked)

    if len(chosen_indices) < count:
        remaining = candidates[~candidates.index.isin(chosen_set)]
        need = min(count - len(chosen_indices), len(remaining))
        if need > 0:
            extra = rng.choice(remaining.index.to_numpy(), size=need, replace=False)
            chosen_indices.extend(extra.tolist())

    sample = candidates.loc[chosen_indices].sort_values("start_time").copy()
    sample["selection_class"] = class_letter
    return sample


def build_selection(catalog: pd.DataFrame, config: SelectionConfig) -> pd.DataFrame:
    samples = []
    for offset, class_letter in enumerate(config.classes):
        sample = uniformly_sample_by_time(
            catalog=catalog,
            class_letter=class_letter,
            count=config.per_class,
            start_date=config.start_date,
            end_date=config.end_date,
            seed=config.seed + offset,
        )
        if len(sample) < config.per_class:
            print(f"Warning: requested {config.per_class} {class_letter}-class flares, selected {len(sample)}")
        samples.append(sample)
    if not samples:
        return pd.DataFrame()
    selected = pd.concat(samples, ignore_index=True)
    return selected.sort_values(["selection_class", "start_time"])


def merge_selection_into_all_flares(selected: pd.DataFrame, state_json_path: Path) -> None:
    all_flares_path = state_json_path.parent / "all_flares.csv"
    all_flares_path.parent.mkdir(parents=True, exist_ok=True)
    if all_flares_path.exists():
        existing = pd.read_csv(all_flares_path)
    else:
        existing = pd.DataFrame()

    merged = pd.concat([existing, selected], ignore_index=True)
    for column in ("start_time", "peak_time", "end_time"):
        if column in merged.columns:
            merged[column] = pd.to_datetime(merged[column], errors="coerce")
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.date
    if "class_value" not in merged.columns and "class" in merged.columns:
        merged["class_value"] = merged["class"].apply(flare_class_to_numeric)
    if "flare_key" not in merged.columns:
        merged["flare_key"] = merged.apply(
            lambda row: build_flare_key(row["start_time"], row["peak_time"], row["end_time"], row["class"]),
            axis=1,
        )
    merged = merged.drop_duplicates(subset=["start_time", "peak_time", "end_time"], keep="first")
    merged = merged.sort_values(["date", "class_value"], ascending=[True, False])
    merged.to_csv(all_flares_path, index=False)
    print(f"Merged selected flares into: {all_flares_path}")


def process_selection(selected: pd.DataFrame, args: argparse.Namespace) -> None:
    overwrite_modules = {"download"} if args.overwrite_download else set()
    if args.overwrite_products:
        overwrite_modules.update({"preprocess", "index"})

    config = PipelineConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        min_flare_class="C1.0",
        state_json_path=args.state_json_path.resolve(),
        data_download_path=args.data_download_path.resolve(),
        run_config=RunConfig(
            existing_data_policy=args.existing_data_policy,
            overwrite_modules=frozenset(overwrite_modules),
        ),
    )

    for flare_date, group in selected.groupby("date"):
        if isinstance(flare_date, str):
            flare_date = date.fromisoformat(flare_date)
        if isinstance(flare_date, pd.Timestamp):
            flare_date = flare_date.date()
        flare_keys = set(group["flare_key"].astype(str))
        print(f"\nProcessing {flare_date}: {len(flare_keys)} selected flares")
        run_download_for_date(config, flare_date)
        run_preprocessing_for_flares(config, flare_keys)
        for idx, flare_key in enumerate(sorted(flare_keys), 1):
            print(f"Index {idx}/{len(flare_keys)}: {flare_key}")
            run_index_calculation_for_flare(config, flare_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select uniformly distributed random C/M flares and process download, maps, indices."
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=date(2008, 1, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--classes", nargs="+", default=["C", "M"])
    parser.add_argument("--per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-json-path", type=Path, default=Path("./data/state.json"))
    parser.add_argument("--data-download-path", type=Path, default=Path("./data"))
    parser.add_argument("--catalog-cache", type=Path, default=Path("./data/random_cm_hek_catalog.csv"))
    parser.add_argument("--selection-file", type=Path, default=Path("./data/random_cm_selection.csv"))
    parser.add_argument("--refresh-catalog", action="store_true")
    parser.add_argument("--use-cache-only", action="store_true")
    parser.add_argument("--chunk-months", type=int, default=1)
    parser.add_argument("--hek-retries", type=int, default=3)
    parser.add_argument("--hek-retry-sleep-seconds", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--existing-data-policy", choices=["skip", "overwrite", "validate"], default="validate")
    parser.add_argument("--overwrite-download", action="store_true")
    parser.add_argument("--overwrite-products", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.classes = tuple(class_name.upper() for class_name in args.classes)
    catalog = load_or_fetch_catalog(args)
    if catalog.empty:
        raise SystemExit("No C/M flares found in HEK catalog")

    selected = build_selection(
        catalog,
        SelectionConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            classes=args.classes,
            per_class=args.per_class,
            seed=args.seed,
        ),
    )
    args.selection_file.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.selection_file, index=False)
    print(f"Saved selection: {args.selection_file}")
    print(selected.groupby("selection_class").size().to_string())

    if args.dry_run:
        print("Dry run complete. No pipeline processing was started.")
        return

    merge_selection_into_all_flares(selected, args.state_json_path)
    process_selection(selected, args)


if __name__ == "__main__":
    main()
