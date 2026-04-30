import csv
from datetime import datetime

import h5py
import numpy as np

from IndexCalculator import IndexCalculator, IndexRegistry, compute_index, retrieve_data


def test_compute_index_returns_nan_when_index_function_raises():
    value = compute_index([], datetime(2025, 1, 1), lambda dates, time_key: (_ for _ in ()).throw(ValueError("boom")))

    assert np.isnan(value)


def test_index_registry_computes_registered_functions_in_order():
    registry = IndexRegistry()
    registry.register("count", lambda dates, time_key: len(dates))
    registry.register("hour", lambda dates, time_key: time_key.hour)

    result = registry.compute_all([(1, 2, 3)], datetime(2025, 1, 1, 5))

    assert result == {"count": 1, "hour": 5}


def test_retrieve_data_reads_only_full_minute_hdf_slices_and_filters_roti(tmp_path):
    h5_path = tmp_path / "map_roti.h5"
    with h5py.File(h5_path, "w") as h5:
        group = h5.create_group("data")
        group.create_dataset(
            "2025-11-11 09:49:00.000000",
            data=np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]]),
        )
        group.create_dataset(
            "2025-11-11 09:49:30.000000",
            data=np.array([[9.0, 9.0, 9.0]]),
        )

    data = retrieve_data(h5_path)

    keys = list(data.keys())
    assert len(keys) == 1
    assert keys[0].replace(tzinfo=None) == datetime(2025, 11, 11, 9, 49)
    assert keys[0].tzinfo is not None
    assert next(iter(data.values())).tolist() == [[3.0, 4.0, 5.0]]


def test_index_file_validation_requires_rows_and_expected_columns(tmp_path):
    calculator = IndexCalculator(base_folder=tmp_path)
    valid = tmp_path / "valid.csv"
    invalid = tmp_path / "invalid.csv"
    valid.write_text("time,day_night_index,gsflai_index,isfai_index\n2025-01-01,0,0,0\n", encoding="utf-8")
    invalid.write_text("time,day_night_index\n", encoding="utf-8")

    assert calculator._is_index_file_valid(valid) is True
    assert calculator._is_index_file_valid(invalid) is False
    assert calculator._is_index_file_valid(tmp_path / "missing.csv") is False


def test_process_single_flare_writes_index_csv(tmp_path):
    flare_key = "20251111T094900_100430_102213_X52"
    map_path = tmp_path / "results" / "X" / "2025-11-11_X52" / "maps" / "map_roti.h5"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(map_path, "w") as h5:
        group = h5.create_group("data")
        group.create_dataset(
            "2025-11-11 09:49:00.000000",
            data=np.array(
                [
                    [0.0, 0.0, 1.0],
                    [10.0, 5.0, 2.0],
                    [0.0, 180.0, 0.5],
                ]
            ),
        )

    calculator = IndexCalculator(base_folder=tmp_path / "results", existing_data_policy="overwrite")
    calculator.registry = IndexRegistry()
    calculator.registry.register("day_night_index", lambda dates, time_key: 1.0)
    calculator.registry.register("gsflai_index", lambda dates, time_key: 2.0)
    calculator.registry.register("isfai_index", lambda dates, time_key: 3.0)

    calculator.process_single_flare(flare_key)

    output = map_path.parent.parent / "indices" / "indices_roti.csv"

    with open(output, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows == [
        {
            "time": "2025-11-11 09:49:00+00:00",
            "day_night_index": "1.0",
            "gsflai_index": "2.0",
            "isfai_index": "3.0",
        }
    ]
