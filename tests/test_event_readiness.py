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
