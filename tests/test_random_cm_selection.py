from datetime import date

import pandas as pd

from scripts.process_random_cm_flares import uniformly_sample_by_time


def test_uniformly_sample_by_time_selects_requested_class_across_time_bins():
    rows = []
    for day in range(10):
        for class_letter in ("C", "M"):
            rows.append(
                {
                    "class": f"{class_letter}1.0",
                    "class_letter": class_letter,
                    "start_time": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                    "flare_key": f"{class_letter}-{day}",
                }
            )
    catalog = pd.DataFrame(rows)

    sample = uniformly_sample_by_time(
        catalog=catalog,
        class_letter="C",
        count=5,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 10),
        seed=1,
    )

    assert len(sample) == 5
    assert set(sample["class_letter"]) == {"C"}
    assert sample["start_time"].is_monotonic_increasing


def test_uniformly_sample_by_time_returns_all_when_less_than_requested():
    catalog = pd.DataFrame(
        [
            {"class": "M1.0", "class_letter": "M", "start_time": pd.Timestamp("2020-01-01"), "flare_key": "a"},
            {"class": "M2.0", "class_letter": "M", "start_time": pd.Timestamp("2020-01-02"), "flare_key": "b"},
        ]
    )

    sample = uniformly_sample_by_time(
        catalog=catalog,
        class_letter="M",
        count=5,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 1, 10),
        seed=1,
    )

    assert list(sample["flare_key"]) == ["a", "b"]
