from pathlib import Path

from results_layout import (
    event_results_dir,
    flare_class_group,
    legacy_event_results_dir,
    product_file_name,
    readable_flare_slug,
    source_file_name,
)


def test_readable_flare_slug_uses_iso_date_and_flare_class_without_time_interval():
    slug = readable_flare_slug("20120307T000200_002400_004000_X54", flare_class="X5.4")

    assert slug == "2012-03-07_X5.4"


def test_flare_class_group_uses_class_letter():
    assert flare_class_group("20120307T000200_002400_004000_X54", flare_class="X5.4") == "X"
    assert flare_class_group("20120307T000200_002400_004000_M12") == "M"
    assert flare_class_group("not-a-flare-key") == "unknown"


def test_public_result_paths_are_stable_for_directory_listing():
    flare_key = "20120307T000200_002400_004000_X54"
    root = event_results_dir(flare_key, flare_class="X5.4", root=Path("results"))

    assert root == Path("results") / "X" / "2012-03-07_X5.4"
    assert source_file_name("goes_xray", flare_key, ".csv", flare_class="X5.4") == "goes_xray.csv"
    assert product_file_name("indices", "dtec_2_10", flare_key, ".csv", flare_class="X5.4") == "indices_dtec_2_10.csv"


def test_legacy_event_results_dir_preserves_old_flat_layout_for_reads():
    flare_key = "20120307T000200_002400_004000_X54"

    assert legacy_event_results_dir(flare_key, flare_class="X5.4", root=Path("results")) == (
        Path("results") / "2012-03-07_X5.4_00-02-00_to_00-40-00"
    )
