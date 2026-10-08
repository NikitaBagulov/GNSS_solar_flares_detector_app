from pathlib import Path

from results_server import (
    breadcrumb_items,
    file_kind,
    format_size,
    graph_product,
    graph_time_label,
    render_directory_html,
    render_dashboard,
    render_event_page,
    render_plot_editor_page,
    render_graph_gallery,
    scan_graph_images,
)


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
    assert 'id="languageSelect"' in html
    assert "Русский" in html


def test_graph_metadata_is_derived_from_plot_filename(tmp_path):
    product_graph = tmp_path / "graphs" / "roti" / "map_roti_01-30-00_UTC.png"
    combined_graph = tmp_path / "graphs" / "combined" / "combined_all-products_01-31-30_UTC.png"
    product_graph.parent.mkdir(parents=True)
    combined_graph.parent.mkdir(parents=True)
    product_graph.write_bytes(b"png")
    combined_graph.write_bytes(b"png")

    assert graph_product(product_graph) == "roti"
    assert graph_product(combined_graph) == "combined"
    assert graph_time_label(product_graph) == "01:30:00"
    assert graph_time_label(combined_graph) == "01:31:30"


def test_render_graph_gallery_includes_interactive_controls(tmp_path):
    graph = tmp_path / "X" / "event" / "graphs" / "dtec_2_10" / "map_dtec_2_10_01-30-00_UTC.png"
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"png")

    images = scan_graph_images(tmp_path, graph.parent.parent)
    html = render_graph_gallery(tmp_path, graph.parent.parent, "/X/event/graphs/").decode("utf-8")

    assert images[0]["product"] == "dtec_2_10"
    assert 'id="mainGraph"' in html
    assert 'id="thumbGrid"' in html
    assert 'id="productFilter"' in html
    assert "/X/event/graphs/dtec_2_10/map_dtec_2_10_01-30-00_UTC.png" in html
    assert "?view=list" in html


def test_render_dashboard_exposes_clear_catalog_filters_and_progress(tmp_path):
    complete = tmp_path / "X" / "2025-11-11_X5.2"
    for product in ("roti", "dtec_2_10", "dtec_10_20", "dtec_20_60"):
        (complete / "maps" / f"map_{product}.h5").parent.mkdir(parents=True, exist_ok=True)
        (complete / "maps" / f"map_{product}.h5").write_bytes(b"map")
        (complete / "indices" / f"indices_{product}.csv").parent.mkdir(parents=True, exist_ok=True)
        (complete / "indices" / f"indices_{product}.csv").write_text("time,value\n", encoding="utf-8")
    graph = complete / "graphs" / "combined" / "preview.png"
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"png")
    (complete / "goes_xray.csv").write_text("time,value\n", encoding="utf-8")
    (complete / "soho_sem.csv").write_text("time,value\n", encoding="utf-8")

    incomplete = tmp_path / "C" / "2025-11-10_C2.0"
    (incomplete / "maps").mkdir(parents=True)

    html = render_dashboard(tmp_path).decode("utf-8")

    assert "Event catalog" in html
    assert "Browse events and see at a glance which data products are ready." in html
    assert 'id="q"' in html
    assert 'id="classFilter"' in html
    assert 'id="statusFilter"' in html
    assert 'id="sortBy"' in html
    assert 'id="resetFilters"' in html
    assert 'id="resultsCount"' in html
    assert 'id="noResults"' in html
    assert 'id="myPlotsForm"' in html
    assert 'id="myPlotsResults"' in html
    assert "Maps 4/4" in html
    assert "Indices 4/4" in html
    assert "GOES available" in html
    assert "SOHO missing" in html
    assert "2025-11-11_X5.2" in html
    assert "2025-11-10_C2.0" in html
    assert "Recently Updated" not in html
    assert 'id="languageSelect"' in html
    assert "Каталог вспышек" in html


def test_render_event_page_groups_preview_status_and_files(tmp_path):
    event = tmp_path / "X" / "2025-11-11_X5.2"
    for product in ("roti", "dtec_2_10", "dtec_10_20", "dtec_20_60"):
        maps_file = event / "maps" / f"map_{product}.h5"
        indices_file = event / "indices" / f"indices_{product}.csv"
        maps_file.parent.mkdir(parents=True, exist_ok=True)
        indices_file.parent.mkdir(parents=True, exist_ok=True)
        maps_file.write_bytes(b"map")
        indices_file.write_text("time,value\n", encoding="utf-8")
    graph = event / "graphs" / "combined" / "preview.png"
    graph.parent.mkdir(parents=True)
    graph.write_bytes(b"png")
    (event / "goes_xray.csv").write_text("time,value\n", encoding="utf-8")
    (event / "soho_sem.csv").write_text("time,value\n", encoding="utf-8")

    html = render_event_page(tmp_path, event).decode("utf-8")

    assert "Back to catalog" in html
    assert "Event details" in html
    assert "Processing status" in html
    assert "Source measurements" in html
    assert "Create a plot" in html
    assert 'href="/editor/X/2025-11-11_X5.2"' in html
    assert 'target="_blank"' in html
    assert 'id="plotPanels"' not in html
    assert 'href="maps/"' in html
    assert 'href="indices/"' in html
    assert "Combined plots" in html
    assert "Browse files" in html
    assert 'id="languageSelect"' in html
    assert "Карты" in html
    assert "Русский" in html


def test_plot_studio_is_separate_page_with_draggable_canvas(tmp_path):
    event = tmp_path / "X" / "2025-11-11_X5.2"
    event.mkdir(parents=True)
    (event / "goes_xray.csv").write_text("time,xrsb\n2025-11-11T01:00:00Z,0.2\n", encoding="utf-8")
    page = render_plot_editor_page(tmp_path, event).decode("utf-8")
    assert 'id="plotCanvas"' in page
    assert 'id="plotStage"' in page
    assert 'id="canvasTitle"' in page
    assert 'id="panelWidth"' in page
    assert 'id="panelHeight"' in page
    assert "setPointerCapture" in page
    assert "layout: 'free'" in page
    assert 'id="plotEmail" type="email" required' in page
    assert 'id="panelEpoch"' in page
    assert 'type="datetime-local"' not in page
    assert "canvas-ylabel" in page and "canvas-xlabel" in page and "canvas-scale" in page
    assert 'href="/X/2025-11-11_X5.2/"' in page
    assert "Русский" in page

    import shutil
    import subprocess
    if shutil.which("node"):
        scripts = page.split("<script>")
        for segment in scripts[1:]:
            script = segment.split("</script>", 1)[0]
            check = subprocess.run(["node", "--check"], input=script, text=True, capture_output=True)
            assert check.returncode == 0, check.stderr
