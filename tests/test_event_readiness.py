import csv

import h5py

from analysis.event_readiness import inspect_event, valid_index, valid_map


def test_valid_map_and_index_require_real_content(tmp_path):
    map_path = tmp_path / "map.h5"
    with h5py.File(map_path, "w") as handle:
        handle.create_group("data").create_dataset("2025-01-01 00:00:00.000000", data=[[0, 0, 1]])

    index_path = tmp_path / "indices.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["time", "day_night_index", "gsflai_index", "isfai_index"],
        )
        writer.writeheader()
        writer.writerow({"time": "2025-01-01", "day_night_index": 1, "gsflai_index": 2, "isfai_index": 3})

    assert valid_map(map_path)
    assert valid_index(index_path)


def test_event_with_map_but_without_index_is_ready_for_index(tmp_path):
    event_dir = tmp_path / "X" / "event"
    maps_dir = event_dir / "maps"
    maps_dir.mkdir(parents=True)
    with h5py.File(maps_dir / "map_roti.h5", "w") as handle:
        handle.create_group("data").create_dataset("2025-01-01 00:00:00.000000", data=[[0, 0, 1]])

    row = inspect_event(
        tmp_path,
        {"name": "event", "path": "X/event", "sources": {"goes_xray": True, "soho_sem": False}},
    )

    assert row["status"] == "ready_for_index"
    assert row["missing_indices"] == "roti"


def _write_index(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["time", "day_night_index", "gsflai_index", "isfai_index"],
        )
        writer.writeheader()
        writer.writerow({"time": "2025-01-01", "day_night_index": 1, "gsflai_index": 2, "isfai_index": 3})


def test_four_indices_are_complete_without_graphs(tmp_path):
    event_dir = tmp_path / "X" / "event"
    for product in ("roti", "dtec_2_10", "dtec_10_20", "dtec_20_60"):
        _write_index(event_dir / "indices" / f"indices_{product}.csv")

    row = inspect_event(tmp_path, {"name": "event", "path": "X/event", "sources": {}})

    assert row["status"] == "complete"
    assert row["qc_decision"] == "include"
    assert row["graphs"] == 0


def test_broken_map_is_reported_without_stopping_inventory(tmp_path):
    event_dir = tmp_path / "X" / "event"
    maps_dir = event_dir / "maps"
    maps_dir.mkdir(parents=True)
    (maps_dir / "map_roti.h5").write_bytes(b"not an HDF5 file")

    row = inspect_event(tmp_path, {"name": "event", "path": "X/event", "sources": {}})

    assert "roti" in row["broken_maps"]
    assert row["status"] == "needs_preprocessing"
