from datetime import date, datetime

import h5py

from DataPreprocessor import DataPreprocessor
from FlareTracker import FlareTracker


class RecordingTracker:
    def __init__(self):
        self.consumed = []

    def mark_source_consumed(self, target_date, source_name, removed_path=None):
        self.consumed.append((target_date, source_name, str(removed_path)))


def _write_valid_h5(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_group("data")


def test_cleanup_consumed_simurg_hdf_deletes_file_and_marks_tracker(tmp_path):
    preprocessor = DataPreprocessor(
        input_root=tmp_path / "data",
        output_dir=tmp_path / "results",
        data_products=["roti"],
    )
    flare = {
        "flare_key": "20251111T094900_100430_102213_X52",
        "class": "X5.2",
        "start_time": datetime(2025, 11, 11, 9, 49),
        "peak_time": datetime(2025, 11, 11, 10, 4, 30),
        "end_time": datetime(2025, 11, 11, 10, 22, 13),
    }
    flare_dir = preprocessor.get_output_dir_for_flare(flare["flare_key"], flare_class=flare["class"])
    _write_valid_h5(preprocessor.get_map_path(flare_dir, flare["flare_key"], "roti", flare_class=flare["class"]))

    source = tmp_path / "data" / "2025-11-11" / "simurg_hdf" / "simurg_hdf_20251111.h5"
    _write_valid_h5(source)
    tracker = RecordingTracker()

    assert preprocessor._maps_available_for_all_flares([flare]) is True
    assert preprocessor._cleanup_consumed_simurg_hdf(source, tracker, date(2025, 11, 11)) is True

    assert not source.exists()
    assert tracker.consumed == [
        (date(2025, 11, 11), "simurg_hdf", str(source)),
    ]


def test_cleanup_consumed_simurg_hdf_ignores_non_simurg_hdf_files(tmp_path):
    preprocessor = DataPreprocessor(input_root=tmp_path / "data", output_dir=tmp_path / "results")
    source = tmp_path / "data" / "2025-11-11" / "other_source" / "other.h5"
    _write_valid_h5(source)

    assert preprocessor._cleanup_consumed_simurg_hdf(source, RecordingTracker(), date(2025, 11, 11)) is False
    assert source.exists()


def test_maps_available_for_all_flares_requires_every_flare_product(tmp_path):
    preprocessor = DataPreprocessor(
        input_root=tmp_path / "data",
        output_dir=tmp_path / "results",
        data_products=["roti"],
    )
    ready_flare = {
        "flare_key": "20251111T094900_100430_102213_X52",
        "class": "X5.2",
        "start_time": datetime(2025, 11, 11, 9, 49),
        "peak_time": datetime(2025, 11, 11, 10, 4, 30),
        "end_time": datetime(2025, 11, 11, 10, 22, 13),
    }
    missing_flare = {
        "flare_key": "20251111T120000_121000_122000_X11",
        "class": "X1.1",
        "start_time": datetime(2025, 11, 11, 12, 0),
        "peak_time": datetime(2025, 11, 11, 12, 10),
        "end_time": datetime(2025, 11, 11, 12, 20),
    }
    flare_dir = preprocessor.get_output_dir_for_flare(
        ready_flare["flare_key"],
        flare_class=ready_flare["class"],
    )
    _write_valid_h5(
        preprocessor.get_map_path(
            flare_dir,
            ready_flare["flare_key"],
            "roti",
            flare_class=ready_flare["class"],
        )
    )

    assert preprocessor._maps_available_for_all_flares([ready_flare]) is True
    assert preprocessor._maps_available_for_all_flares([ready_flare, missing_flare]) is False


def test_flare_tracker_treats_consumed_source_as_available():
    tracker = FlareTracker.__new__(FlareTracker)
    tracker.state = {"consumed_sources_by_date": {}}

    assert tracker._check_source_has_data("simurg_hdf", date(2025, 11, 11)) is False

    tracker.state["consumed_sources_by_date"]["2025-11-11"] = {
        "simurg_hdf": {"removed_path": "data/2025-11-11/simurg_hdf/file.h5"}
    }

    assert tracker._check_source_has_data("simurg_hdf", date(2025, 11, 11)) is True
