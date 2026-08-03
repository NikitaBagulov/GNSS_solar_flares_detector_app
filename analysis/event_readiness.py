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
STATUSES = ("complete", "ready_for_index", "needs_preprocessing")


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


def _map_state(path: Path) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        return "missing"
    return "valid" if valid_map(path) else "broken"


def _euv_metadata_state(event_dir: Path, has_euv: bool) -> str:
    if not has_euv:
        return "source_missing"
    candidates = (
        event_dir / "soho_sem" / "metadata.json",
        event_dir / "soho_sem_metadata.json",
        event_dir / "metadata" / "soho_sem.json",
    )
    return "documented" if any(path.is_file() for path in candidates) else "metadata_missing"


def inspect_event(results_dir: Path, event: dict) -> dict[str, object]:
    event_dir = results_dir / event["path"]
    map_states = {
        product: _map_state(event_dir / "maps" / f"map_{product}.h5")
        for product in PRODUCTS
    }
    map_products = [product for product, state in map_states.items() if state == "valid"]
    index_products = [p for p in PRODUCTS if valid_index(event_dir / "indices" / f"indices_{p}.csv")]
    missing_indices = [p for p in PRODUCTS if p not in index_products]
    graph_count = sum(1 for path in (event_dir / "graphs").rglob("*.png") if path.is_file())

    # Scientific readiness is defined by the four index tables. Graphs are
    # derived presentation products and therefore never block inclusion.
    if len(index_products) == len(PRODUCTS):
        status = "complete"
        decision = "include"
        reason = "all_four_indices_valid"
    elif any(map_states[p] == "valid" for p in missing_indices):
        status = "ready_for_index"
        decision = "exclude"
        reason = "indices_missing_but_maps_available"
    else:
        status = "needs_preprocessing"
        decision = "exclude"
        reason = "indices_missing_and_maps_unavailable"

    has_euv = bool(event.get("sources", {}).get("soho_sem"))
    return {
        "event": event["name"],
        "path": event["path"],
        "status": status,
        "qc_decision": decision,
        "qc_reason": reason,
        "maps": len(map_products),
        "indices": len(index_products),
        "graphs": graph_count,
        "goes": bool(event.get("sources", {}).get("goes_xray")),
        "soho_sem": has_euv,
        "euv_metadata": _euv_metadata_state(event_dir, has_euv),
        "missing_maps": " ".join(p for p, state in map_states.items() if state == "missing"),
        "broken_maps": " ".join(p for p, state in map_states.items() if state == "broken"),
        "missing_indices": " ".join(missing_indices),
    }


def inventory(results_dir: Path) -> list[dict[str, object]]:
    sys.path.insert(0, str(REPO_ROOT))
    from results_server import scan_events

    return [inspect_event(results_dir, event) for event in scan_events(results_dir)]


def print_table(rows: list[dict[str, object]]) -> None:
    columns = ("event", "status", "qc_decision", "qc_reason", "maps", "indices", "graphs", "goes", "soho_sem", "euv_metadata", "missing_maps", "broken_maps", "missing_indices")
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
