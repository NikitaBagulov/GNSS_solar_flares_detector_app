"""Inventory computed flare events and identify cheap next pipeline steps.

Examples:
    python3 -m analysis.event_readiness --results-dir results
    python3 -m analysis.event_readiness --results-dir results --status ready_for_index --format events
    python3 -m analysis.event_readiness --results-dir results --output readiness.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ("roti", "dtec_2_10", "dtec_10_20", "dtec_20_60")
INDEX_COLUMNS = {"time", "day_night_index", "gsflai_index", "isfai_index"}
STATUSES = ("complete", "ready_for_plots", "ready_for_index", "needs_preprocessing")


def valid_map(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with h5py.File(path, "r") as handle:
            return "data" in handle and len(handle["data"]) > 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"WARNING: invalid HDF5 map {path}: {exc}", file=sys.stderr)
        return False


def valid_index(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            return INDEX_COLUMNS.issubset(set(reader.fieldnames or ())) and next(reader, None) is not None
    except (OSError, csv.Error, UnicodeError):
        return False


def inspect_event(results_dir: Path, event: dict) -> dict[str, object]:
    event_dir = results_dir / event["path"]
    map_products = [p for p in PRODUCTS if valid_map(event_dir / "maps" / f"map_{p}.h5")]
    index_products = [p for p in PRODUCTS if valid_index(event_dir / "indices" / f"indices_{p}.csv")]
    missing_indices = [p for p in map_products if p not in index_products]
    graph_count = sum(1 for path in (event_dir / "graphs").rglob("*.png") if path.is_file())

    if len(map_products) == len(PRODUCTS) and len(index_products) == len(PRODUCTS) and graph_count:
        status = "complete"
    elif index_products and graph_count == 0:
        status = "ready_for_plots"
    elif missing_indices:
        status = "ready_for_index"
    else:
        status = "needs_preprocessing"

    return {
        "event": event["name"],
        "path": event["path"],
        "status": status,
        "maps": len(map_products),
        "indices": len(index_products),
        "graphs": graph_count,
        "goes": bool(event.get("sources", {}).get("goes_xray")),
        "soho_sem": bool(event.get("sources", {}).get("soho_sem")),
        "missing_indices": " ".join(missing_indices),
    }


def inventory(results_dir: Path) -> list[dict[str, object]]:
    sys.path.insert(0, str(REPO_ROOT))
    from results_server import scan_events

    return [inspect_event(results_dir, event) for event in scan_events(results_dir)]


def print_table(rows: list[dict[str, object]]) -> None:
    columns = ("event", "status", "maps", "indices", "graphs", "goes", "soho_sem", "missing_indices")
    widths = {column: max(len(column), *(len(str(row[column])) for row in rows)) for column in columns}
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect event products already present under results/.")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--status", choices=STATUSES)
    parser.add_argument("--format", choices=("table", "events"), default="table")
    parser.add_argument("--output", type=Path, help="Optional CSV inventory path")
    args = parser.parse_args(argv)

    rows = inventory(args.results_dir)
    if args.status:
        rows = [row for row in rows if row["status"] == args.status]

    if args.output and rows:
        write_csv(rows, args.output)
    if args.format == "events":
        print(" ".join(str(row["event"]) for row in rows))
    elif rows:
        print_table(rows)
    else:
        print("No matching events found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
