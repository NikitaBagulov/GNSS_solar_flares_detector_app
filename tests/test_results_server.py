from pathlib import Path

from results_server import breadcrumb_items, file_kind, format_size, render_directory_html


def test_format_size_uses_readable_units():
    assert format_size(None) == "-"
    assert format_size(12) == "12 B"
    assert format_size(2048) == "2.0 KB"
    assert format_size(5 * 1024 * 1024) == "5.0 MB"


def test_file_kind_labels_known_files_and_folders(tmp_path):
    folder = tmp_path / "graphs"
    folder.mkdir()
    csv_file = tmp_path / "goes_xray.csv"
    csv_file.write_text("time,xrsb\n", encoding="utf-8")
    unknown = tmp_path / "artifact.bin"
    unknown.write_bytes(b"data")

    assert file_kind(folder) == "Folder"
    assert file_kind(csv_file) == "CSV"
    assert file_kind(unknown) == "BIN"


def test_breadcrumb_items_build_clickable_path():
    assert breadcrumb_items("/X/2025-11-11_X5.2/graphs/") == [
        ("Results", "/"),
        ("X", "/X/"),
        ("2025-11-11_X5.2", "/X/2025-11-11_X5.2/"),
        ("graphs", "/X/2025-11-11_X5.2/graphs/"),
    ]


def test_render_directory_html_lists_folders_before_files(tmp_path):
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    csv_file = tmp_path / "goes_xray.csv"
    csv_file.write_text("time,xrsb\n", encoding="utf-8")

    html = render_directory_html("/", [csv_file, graphs]).decode("utf-8")

    assert "<h1>results</h1>" in html
    assert "graphs/" in html
    assert "goes_xray.csv" in html
    assert html.index("graphs/") < html.index("goes_xray.csv")
