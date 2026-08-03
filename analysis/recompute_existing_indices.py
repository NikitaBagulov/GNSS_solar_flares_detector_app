"""Recompute indices directly from map HDF5 files already stored in results/."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from IndexCalculator import IndexCalculator, retrieve_data  # noqa: E402
from results_server import PRODUCTS, scan_events  # noqa: E402


def selected_events(results_dir: Path, names: list[str] | None) -> list[dict]:
    events = scan_events(results_dir)
    if not names:
        return events
    wanted = set(names)
    return [event for event in events if event["name"] in wanted or event["path"] in wanted]


def compute_map(calculator: IndexCalculator, map_path: Path) -> list[dict]:
    rows = []
    for time_key, array in retrieve_data(map_path).items():
        points = [tuple(row) for row in array]
        values = calculator.registry.compute_all(points, time_key)
        values["time"] = time_key
        rows.append(values)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["time", "day_night_index", "gsflai_index", "isfai_index"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--events", nargs="*", help="Event names or paths; default is every event")
    parser.add_argument("--products", nargs="+", choices=PRODUCTS, default=list(PRODUCTS))
    parser.add_argument("--policy", choices=("skip", "validate", "overwrite"), default="validate")
    parser.add_argument("--backup-existing", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    calculator = IndexCalculator(base_folder=args.results_dir, existing_data_policy=args.policy)
    events = selected_events(args.results_dir, args.events)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_rows = []

    print(f"Events selected: {len(events)}")
    for event_number, event in enumerate(events, 1):
        event_dir = args.results_dir / event["path"]
        print(f"[{event_number}/{len(events)}] {event['name']}")
        for product in args.products:
            map_path = event_dir / "maps" / f"map_{product}.h5"
            output_path = event_dir / "indices" / f"indices_{product}.csv"

            if not map_path.exists():
                status = "missing_map"
            elif output_path.exists() and args.policy == "skip":
                status = "skipped_existing"
            elif output_path.exists() and args.policy == "validate" and calculator._is_index_file_valid(output_path):
                status = "skipped_valid"
            else:
                if output_path.exists() and args.backup_existing:
                    backup = event_dir / f"indices_legacy_{stamp}" / output_path.name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(output_path, backup)
                try:
                    rows = compute_map(calculator, map_path)
                    if not rows:
                        status = "no_rows"
                    else:
                        write_rows(output_path, rows)
                        status = f"computed:{len(rows)}"
                except Exception as exc:  # keep the batch running and record the failure
                    status = f"error:{type(exc).__name__}:{exc}"

            print(f"  {product}: {status}")
            report_rows.append({"event": event["name"], "product": product, "status": status})

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["event", "product", "status"])
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"Saved report: {args.report}")

    failures = [row for row in report_rows if str(row["status"]).startswith("error:")]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
