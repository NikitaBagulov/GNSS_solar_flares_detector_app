from datetime import datetime
import json

import h5py
import numpy as np
import pandas as pd
from dateutil import tz

from PlotDataLoader import PlotDataLoader


def test_load_maps_skips_unreadable_hdf5_product(tmp_path):
    valid_path = tmp_path / "map_roti.h5"
    broken_path = tmp_path / "map_dtec_2_10.h5"

    with h5py.File(valid_path, "w") as h5:
        group = h5.create_group("data")
        group.create_dataset(
            "2025-11-11 09:49:00.000000",
            data=np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]]),
        )
    broken_path.write_bytes(b"not an hdf5 file")

    loader = PlotDataLoader.__new__(PlotDataLoader)
    start = datetime(2025, 11, 11, 9, 48, tzinfo=tz.gettz("UTC"))
    end = datetime(2025, 11, 11, 9, 50, tzinfo=tz.gettz("UTC"))

    timestamps, product_values = loader._load_maps(
        {
            "dtec_2_10": broken_path,
            "roti": valid_path,
        },
        start,
        end,
    )

    assert [time.replace(tzinfo=None) for time in timestamps] == [
        datetime(2025, 11, 11, 9, 49)
    ]
    assert product_values[0]["roti"].tolist() == [[3.0, 4.0, 5.0]]
    assert product_values[0]["dtec_2_10"].size == 0


def test_load_flare_keeps_index_times_per_product(tmp_path):
    flare_key = "20251111T094900_100430_102213_X52"
    flares_file = tmp_path / "flares.csv"
    state_file = tmp_path / "state.json"
    roti_indices = tmp_path / "indices_roti.csv"
    dtec_indices = tmp_path / "indices_dtec_2_10.csv"

    pd.DataFrame(
        [
            {
                "flare_key": flare_key,
                "start_time": "2025-11-11T09:49:00Z",
                "peak_time": "2025-11-11T10:04:30Z",
                "end_time": "2025-11-11T10:22:13Z",
                "class": "X5.2",
                "hpc_x": 1.0,
                "hpc_y": 2.0,
            }
        ]
    ).to_csv(flares_file, index=False)

    pd.DataFrame(
        {
            "time": ["2025-11-11T09:49:00Z", "2025-11-11T09:50:00Z"],
            "day_night_index": [1.0, 2.0],
            "gsflai_index": [3.0, 4.0],
            "isfai_index": [5.0, 6.0],
        }
    ).to_csv(roti_indices, index=False)
    pd.DataFrame(
        {
            "time": ["2025-11-11T09:49:00Z"],
            "day_night_index": [7.0],
            "gsflai_index": [8.0],
            "isfai_index": [9.0],
        }
    ).to_csv(dtec_indices, index=False)

    state_file.write_text(
        json.dumps(
            {
                "files_by_flare": {
                    flare_key: {
                        "maps": {},
                        "indices": {
                            "roti": str(roti_indices),
                            "dtec_2_10": str(dtec_indices),
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    data = PlotDataLoader(str(flares_file), str(state_file)).load_flare(flare_key)

    assert len(data.index_times_by_product["roti"]) == 2
    assert len(data.indices["roti"]["day_night_index"]) == 2
    assert len(data.index_times_by_product["dtec_2_10"]) == 1
    assert len(data.indices["dtec_2_10"]["day_night_index"]) == 1
