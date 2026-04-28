from pathlib import Path

from results_layout import event_results_dir, product_file_name, readable_flare_slug, source_file_name


def test_readable_flare_slug_uses_iso_date_and_event_times():
    slug = readable_flare_slug("20120307T000200_002400_004000_X54", flare_class="X5.4")

    assert slug == "2012-03-07_X5.4_00-02-00_to_00-40-00"


def test_public_result_paths_are_stable_for_directory_listing():
    flare_key = "20120307T000200_002400_004000_X54"
    root = event_results_dir(flare_key, flare_class="X5.4", root=Path("results"))

    assert root == Path("results") / "2012-03-07_X5.4_00-02-00_to_00-40-00"
    assert source_file_name("goes_xray", flare_key, ".csv", flare_class="X5.4") == "goes_xray.csv"
    assert product_file_name("indices", "dtec_2_10", flare_key, ".csv", flare_class="X5.4") == "indices_dtec_2_10.csv"
