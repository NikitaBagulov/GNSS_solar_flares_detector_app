from datetime import datetime

import h5py
import numpy as np
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
