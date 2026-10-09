import json
import io
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import h5py
import numpy as np
import pytest
from PIL import Image

from plot_requests import SERIES, SIMPLE_COLORS, _flare_window, cleanup_cache, figure_caption, map_epochs, render_plot, validate_request, _map_points
from flare_metadata import flare_metadata, scientific_caption
from results_server import PrettyDirectoryHandler, render_plot_editor_page


@pytest.fixture
def event(tmp_path):
    path = tmp_path / "X" / "2025-11-11_X5.2"
    (path / "indices").mkdir(parents=True)
    (path / "indices" / "indices_roti.csv").write_text(
        "time,day_night_index,gsflai_index,isfai_index\n"
        "2025-11-11T01:00:00Z,0.5,0.2,0.1\n"
        "2025-11-11T01:01:00Z,0.8,0.3,0.2\n", encoding="utf-8"
    )
    (path / "goes_xray.csv").write_text("time,xrsb\n2025-11-11T01:00:00Z,0.001\n", encoding="utf-8")
    (path / "maps").mkdir()
    with h5py.File(path / "maps" / "map_roti.h5", "w") as file:
        file.create_dataset("data/2025-11-11 01:00:00.000000", data=np.array(
            [(51.0, 30.0, 0.7)], dtype=[("lat", "f4"), ("lon", "f4"), ("vals", "f4")]
        ))
    return path


def test_plot_renders_selected_panels_and_validates_inputs(event):
    request = {"panels": [{"series": "roti:gsflai_index", "color": "#123456"},
                          {"series": "map:roti", "color": "#2878a5"}], "layout": "grid", "title": "My plot",
               "email": " Person@Example.org "}
    assert render_plot(event, request).startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ValueError, match="Unknown"):
        validate_request(event, {"panels": [{"series": "../../secret"}], "email": "person@example.org"})
    with pytest.raises(ValueError, match="email"):
        validate_request(event, {"panels": [{"series": "goes"}], "email": "invalid"})
    with pytest.raises(ValueError, match="email"):
        validate_request(event, {"panels": [{"series": "goes"}]})


def test_freeform_panels_render_at_canvas_aspect_ratio_and_validate_bounds(event):
    request = {"layout": "free", "email": "person@example.org", "panels": [
        {"series": "goes", "color": "#d1495b", "rect": {"x": .04, "y": .04, "w": .66, "h": .3}},
        {"series": "map:roti", "rect": {"x": .3, "y": .46, "w": .62, "h": .48}},
    ]}
    with Image.open(io.BytesIO(render_plot(event, request))) as image:
        assert abs(image.width / image.height - 12 / 8.5) < .01
        pixels = np.asarray(image.convert("RGB"))
        # A PNG containing only axes and panel backgrounds used to pass this test.
        assert np.count_nonzero((pixels[:, :, 0] > pixels[:, :, 1] * 1.3) &
                                (pixels[:, :, 0] > pixels[:, :, 2] * 1.1)) > 20
        assert np.count_nonzero((pixels[:, :, 1] > pixels[:, :, 0] * 1.2) &
                                (pixels[:, :, 1] > pixels[:, :, 2] * 1.1)) > 20
    request["panels"][1]["rect"]["w"] = .8
    with pytest.raises(ValueError, match="canvas"):
        validate_request(event, request)
    request["panels"][1]["rect"]["w"] = float("nan")
    with pytest.raises(ValueError, match="position"):
        validate_request(event, request)


def test_freeform_rejects_overlapping_panels_and_renders_sun(event):
    request = {"layout": "free", "email": "person@example.org", "panels": [
        {"series": "sun", "rect": {"x": .04, "y": .04, "w": .44, "h": .55}},
        {"series": "goes", "rect": {"x": .52, "y": .04, "w": .44, "h": .55}},
    ]}
    with Image.open(io.BytesIO(render_plot(event, request))) as image:
        pixels = np.asarray(image.convert("RGB"))
        assert np.count_nonzero((pixels[:, :, 0] > 200) & (pixels[:, :, 1] > 130) & (pixels[:, :, 2] < 135)) > 100
    request["panels"][1]["rect"]["x"] = .4
    with pytest.raises(ValueError, match="overlap"):
        validate_request(event, request)


def test_simple_and_plotter_styles_change_render_and_validate_choice(event):
    (event / "soho_sem.csv").write_text(
        "time,flux_01_50\n2025-11-11T01:00:00Z,100\n2025-11-11T01:01:00Z,120\n", encoding="utf-8")
    request = {"layout": "free", "email": "user@example.org", "panels": [
        {"series": "goes", "rect": {"x": .04, "y": .04, "w": .9, "h": .38}},
        {"series": "soho", "rect": {"x": .04, "y": .52, "w": .9, "h": .38}},
    ]}
    assert validate_request(event, request)["style"] == "plotter"
    colors = {}
    for style in ("simple", "plotter"):
        request["style"] = style
        with Image.open(io.BytesIO(render_plot(event, request))) as image:
            pixels = np.asarray(image.convert("RGB"))
            colors[style] = pixels
    assert np.all(colors["simple"][0, 0] == 255)
    assert not np.all(colors["plotter"][0, 0] == 255)
    simple = colors["simple"]
    assert np.count_nonzero((simple[:, :, 1] > simple[:, :, 0] * 1.2) &
                            (simple[:, :, 1] > simple[:, :, 2] * 1.2)) > 20
    plotter = colors["plotter"]
    assert np.count_nonzero((plotter[:, :, 0] > plotter[:, :, 1] * 1.3) &
                            (plotter[:, :, 0] > plotter[:, :, 2] * 1.1)) > 20
    request["style"] = "not-a-style"
    with pytest.raises(ValueError, match="style"):
        validate_request(event, request)


def test_print_width_chart_types_flux_units_and_shared_axes(event, monkeypatch):
    from matplotlib.figure import Figure

    (event / "goes_xray.csv").write_text(
        "time,xrsa,xrsb\n2025-11-11T01:00:00Z,0.00001,0.0001\n"
        "2025-11-11T01:01:00Z,0.00002,0.0003\n", encoding="utf-8")
    panels = [{"series": "goes", "plot_type": "bar", "rect": {"x": .03, "y": .03, "w": .94, "h": .40}},
              {"series": "goes", "plot_type": "line", "fill_negative": True,
               "rect": {"x": .03, "y": .55, "w": .94, "h": .40}}]
    request = {"email": "test@example.org", "layout": "free", "style": "simple", "panels": panels,
               "flux_mode": "relative", "shared_y": True, "print_size": "one-column"}
    original = Figure.savefig
    inspected = []

    def inspect(fig, *args, **kwargs):
        assert fig.get_size_inches().tolist() == pytest.approx([3.5, 4.95])
        fig.canvas.draw()
        for label in fig.texts:
            bounds = label.get_window_extent(fig.canvas.get_renderer())
            assert bounds.x0 >= -1 and bounds.x1 <= fig.bbox.x1 + 1
            assert bounds.y0 >= -1 and bounds.y1 <= fig.bbox.y1 + 1
        axes = [ax for ax in fig.axes if ax.get_xlabel() == "Time (UTC)"]
        assert len(axes) == 2
        assert axes[0].patches and not axes[1].patches
        assert axes[1].collections  # Explicit negative fill, even if this event never drops below zero.
        assert axes[0].get_ylim() == pytest.approx(axes[1].get_ylim())
        inspected.append(True)
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    with Image.open(io.BytesIO(render_plot(event, request))) as image:
        assert image.size == (1050, 1485)
    assert inspected
    assert "(A) GOES X-ray flux; (B) GOES X-ray flux" in figure_caption(event, request)
    request["flux_mode"] = "physical"
    request["panels"] = panels[:1]
    assert validate_request(event, request)["flux_mode"] == "physical"
    request["panels"][0]["plot_type"] = "heatmap"
    with pytest.raises(ValueError, match="plot type"):
        validate_request(event, request)
    request["panels"][0]["plot_type"] = "auto"
    request["panels"] = panels + [{"series": "sun", "rect": {"x": .01, "y": .01, "w": .20, "h": .20}}]
    with pytest.raises(ValueError, match="at most two panels"):
        validate_request(event, request)


def test_draft_endpoint_uses_real_data_without_storing_plot(event, tmp_path):
    handler = partial(PrettyDirectoryHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        payload = json.dumps({"event": "X/2025-11-11_X5.2", "layout": "free", "print_size": "one-column",
                              "panels": [{"series": "goes", "rect": {"x": .04, "y": .05, "w": .92, "h": .88}}]}).encode()
        with urlopen(Request(f"http://127.0.0.1:{server.server_port}/api/plots/preview", data=payload)) as response:
            assert response.headers["Content-Type"] == "image/png"
            with Image.open(io.BytesIO(response.read())) as image:
                assert image.width == 315
        assert not (tmp_path / ".plot-cache").exists()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_publication_labels_and_focused_flare_window(event, monkeypatch):
    from datetime import datetime
    from matplotlib.figure import Figure
    import flare_metadata as catalog

    (event / "goes_xray.csv").write_text(
        "time,xrsb\n2025-11-11T01:00:00Z,0.001\n2025-11-11T12:00:00Z,0.9\n", encoding="utf-8")
    (event / "solar_image").mkdir()
    (event / "solar_image" / "corrupt.png").write_bytes(b"not an image")
    metadata = {"class": "X5.2", "date": "2025-11-11", "start": "2025-11-11T01:00:00+00:00",
                "peak": "2025-11-11T01:05:00+00:00", "end": "2025-11-11T01:10:00+00:00",
                "x": None, "y": None}
    monkeypatch.setattr(catalog, "flare_metadata", lambda _: metadata)
    assert _flare_window(metadata) == (datetime(2025, 11, 11, 0, 50), datetime(2025, 11, 11, 1, 20))
    observed = []
    original_savefig = Figure.savefig

    def inspect_figure(fig, *args, **kwargs):
        assert not fig.patches  # Editor panel outlines do not belong in the exported figure.
        titles = [text.get_text() for text in fig.texts]
        assert any("Global ROTI map · 01:00 UTC" == text for text in titles)
        assert titles.count("Solar disk") == 1
        assert SERIES["goes"] in titles
        sun_axes = [axis for axis in fig.axes if axis.patches and
                    any(type(patch).__name__ == "Circle" for patch in axis.patches)]
        assert sun_axes and sun_axes[0].get_title(loc="left") == ""
        goes_axes = [axis for axis in fig.axes if axis.lines and
                     any(line.get_color() == SIMPLE_COLORS["goes"] for line in axis.lines)]
        assert goes_axes
        ax = goes_axes[0]
        assert max(ax.lines[0].get_ydata()) < .01  # The 12:00 spike was cropped before autoscaling.
        assert len(ax.lines) == 3  # X-ray flux, peak and selected time.
        assert ax.lines[-1].get_linestyle() == ":"
        assert [text.get_text() for text in fig.legends[0].get_texts()] == [
            "Flare peak", "Selected time 01:00 UTC"]
        assert "2025-11-11 X5.2 Solar Flare" in titles
        assert any("Peak: 2025-11-11 01:05 UTC" in text for text in titles)
        observed.append(True)
        return original_savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    request = {"layout": "free", "style": "simple", "email": "user@example.org", "panels": [
        {"series": "map:roti", "rect": {"x": .03, "y": .03, "w": .53, "h": .44}},
        {"series": "sun", "rect": {"x": .58, "y": .03, "w": .39, "h": .44}},
        {"series": "goes", "rect": {"x": .03, "y": .53, "w": .94, "h": .41}},
    ]}
    assert render_plot(event, request).startswith(b"\x89PNG")
    assert observed


def test_available_goes_and_soho_channels_are_labelled_without_inventing_missing_channels(event, monkeypatch):
    from matplotlib.figure import Figure
    import flare_metadata as catalog

    (event / "goes_xray.csv").write_text(
        "time,xrsa,xrsb\n2025-11-11T01:00:00Z,0.00001,0.0001\n"
        "2025-11-11T01:05:00Z,0.0001,0.001\n", encoding="utf-8")
    (event / "soho_sem.csv").write_text(
        "time,flux_26_34,flux_01_50\n2025-11-11T01:00:00Z,100,200\n"
        "2025-11-11T01:05:00Z,150,300\n", encoding="utf-8")
    monkeypatch.setattr(catalog, "flare_metadata", lambda _: {
        "class": "X5.2", "date": "2025-11-11", "start": "2025-11-11T01:00:00+00:00",
        "peak": "2025-11-11T01:00:00+00:00", "end": "2025-11-11T01:10:00+00:00",
        "x": None, "y": None})
    original = Figure.savefig
    inspected = []

    def inspect(fig, *args, **kwargs):
        time_axes = [axis for axis in fig.axes if axis.get_xlabel() == "Time (UTC)"]
        assert len(time_axes) == 2
        goes, soho = time_axes
        assert goes.get_yscale() == soho.get_yscale() == "linear"
        assert goes.get_ylabel() == soho.get_ylabel() == "Flux change (%)"
        assert goes.lines[0].get_ydata() == pytest.approx([0, 900])
        assert soho.lines[0].get_ydata() == pytest.approx([0, 50])
        assert goes.get_position().height > .40 * .82 * .6
        assert [line.get_label() for line in goes.lines[:2]] == [
            "GOES XRS-A (0.05–0.4 nm)", "GOES XRS-B (0.1–0.8 nm)"]
        assert [line.get_label() for line in soho.lines[:2]] == [
            "SOHO/SEM 26–34 nm", "SOHO/SEM 0.1–50 nm"]
        assert [line.get_linestyle() for line in soho.lines[:2]] == ["--", "-"]
        assert [line.get_color() for line in soho.lines[:2]] == ["#333333", "#006400"]
        assert len(goes.lines) == len(soho.lines) == 3  # Shared peak/selected epoch.
        assert fig.legends == []
        inspected.append(True)
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    request = {"layout": "free", "style": "simple", "email": "user@example.org", "panels": [
        {"series": "goes", "rect": {"x": .02, "y": .04, "w": .96, "h": .40}},
        {"series": "soho", "rect": {"x": .02, "y": .52, "w": .96, "h": .40}}]}
    assert render_plot(event, request).startswith(b"\x89PNG")
    assert inspected


def test_colorbar_matches_rendered_map_height(event, monkeypatch):
    from matplotlib.figure import Figure

    original = Figure.savefig

    def inspect(fig, *args, **kwargs):
        fig.canvas.draw()
        map_axes = next(axis for axis in fig.axes if axis.child_axes)
        map_box = map_axes.get_position()
        bar_box = map_axes.child_axes[0].get_position()
        assert bar_box.y0 == pytest.approx(map_box.y0, abs=1e-6)
        assert bar_box.y1 == pytest.approx(map_box.y1, abs=1e-6)
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    assert render_plot(event, {"layout": "free", "style": "simple", "email": "user@example.org",
                               "panels": [{"series": "map:roti", "rect": {"x": .02, "y": .02,
                                                                           "w": .58, "h": .42}}]}).startswith(b"\x89PNG")


@pytest.mark.parametrize("style", ["simple", "plotter"])
def test_map_color_limits_are_fixed_for_roti_and_dtec(event, monkeypatch, style):
    from matplotlib.figure import Figure
    from Plotter import CombinedPlotter

    assert CombinedPlotter._get_product_color_range("roti") == (0, .5)
    assert CombinedPlotter._get_product_color_range("dtec_2_10") == (-.5, .5)
    stamp = "2025-11-11 01:00:00.000000"
    with h5py.File(event / "maps" / "map_dtec_2_10.h5", "w") as file:
        file.create_dataset(f"data/{stamp}", data=np.array(
            [(51., 30., -.8)], dtype=[("lat", "f4"), ("lon", "f4"), ("vals", "f4")]))

    original = Figure.savefig

    def inspect(fig, *args, **kwargs):
        maps = [axis for axis in fig.axes if axis.child_axes]
        assert len(maps) == 2
        assert [(ax.collections[0].norm.vmin, ax.collections[0].norm.vmax) for ax in maps] == [
            (0, .5), (-.5, .5)]
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    request = {"layout": "free", "style": style, "email": "user@example.org", "epoch": stamp,
               "panels": [{"series": "map:roti", "rect": {"x": .02, "y": .02, "w": .45, "h": .60}},
                          {"series": "map:dtec_2_10", "rect": {"x": .52, "y": .02, "w": .45, "h": .60}}]}
    assert render_plot(event, request).startswith(b"\x89PNG")


def test_each_plot_title_is_centered_in_free_and_vertical_layouts(event, monkeypatch):
    from matplotlib.figure import Figure

    original = Figure.savefig
    layout = "free"
    rects = [{"x": .02, "y": .02, "w": .45, "h": .40},
             {"x": .52, "y": .02, "w": .45, "h": .40},
             {"x": .02, "y": .52, "w": .96, "h": .40}]

    def inspect(fig, *args, **kwargs):
        if layout == "free":
            for title, rect in zip(("GOES X-ray flux", "Solar disk", "Global ROTI map"), rects):
                label = next(text for text in fig.texts if text.get_text().startswith(title))
                assert label.get_position()[0] == pytest.approx(.055 + .89 * (rect["x"] + rect["w"] / 2))
                assert label.get_ha() == "center"
        else:
            assert [ax.get_title(loc="center") for ax in fig.axes] == [
                "GOES X-ray flux", "Solar disk", "ROTI map · 2025-11-11 01:00 UTC"]
            assert all(not ax.get_title(loc="left") for ax in fig.axes)
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    panels = [{"series": key, "rect": rect} for key, rect in zip(("goes", "sun", "map:roti"), rects)]
    for layout in ("free", "vertical"):
        assert render_plot(event, {"layout": layout, "style": "simple", "email": "user@example.org",
                                   "panels": panels}).startswith(b"\x89PNG")


def test_plotter_style_marks_flare_and_map_time_and_uses_compact_flux_axes(event, monkeypatch):
    from matplotlib.figure import Figure
    import flare_metadata as catalog

    metadata = {"class": "X5.2", "date": "2025-11-11", "start": "2025-11-11T01:00:00+00:00",
                "peak": "2025-11-11T01:05:00+00:00", "end": "2025-11-11T01:10:00+00:00",
                "x": None, "y": None}
    monkeypatch.setattr(catalog, "flare_metadata", lambda _: metadata)
    saved = Figure.savefig
    observed = []

    def inspect(fig, *args, **kwargs):
        ax = next(axis for axis in fig.axes if axis.lines and axis.lines[0].get_color() == "#d1495b")
        assert len(ax.lines) == 3
        assert ax.lines[-1].get_color() == "#9a6400"
        assert ax.get_position().x0 < .055 + .04 * .89 + .16 * (.9 * .89)
        assert "Selected time 01:00 UTC" in [text.get_text() for text in fig.legends[0].get_texts()]
        observed.append(True)
        return saved(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    request = {"layout": "free", "style": "plotter", "email": "user@example.org", "panels": [
        {"series": "goes", "rect": {"x": .04, "y": .04, "w": .9, "h": .38}},
        {"series": "map:roti", "rect": {"x": .04, "y": .52, "w": .9, "h": .38}},
    ]}
    assert render_plot(event, request).startswith(b"\x89PNG")
    assert observed


def test_flare_caption_uses_catalog_metadata_and_does_not_invent_position(event, tmp_path, monkeypatch):
    import flare_metadata as catalog
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "all_flares.csv").write_text(
        "class,start_time,peak_time,end_time,hpc_x,hpc_y\n"
        "X5.2,2025-11-11T01:00:00Z,2025-11-11T01:05:00Z,2025-11-11T01:10:00Z,100,-200\n", encoding="utf-8")
    monkeypatch.setattr(catalog, "__file__", str(tmp_path / "flare_metadata.py"))
    metadata = flare_metadata(event)
    assert metadata["x"] == 100 and metadata["y"] == -200
    heading, subtitle = scientific_caption(metadata)
    assert heading == "2025-11-11 X5.2 Solar Flare"
    assert subtitle == "Peak: 2025-11-11 01:05 UTC"
    (tmp_path / "data" / "all_flares.csv").write_text(
        "class,start_time,peak_time,end_time,hpc_x,hpc_y\n"
        "X5.2,2025-11-11T01:00:00Z,2025-11-11T01:05:00Z,2025-11-11T01:10:00Z,,\n", encoding="utf-8")
    assert flare_metadata(event)["x"] is None


def test_nearby_catalog_class_requires_unique_peak_in_event_maps(event, tmp_path, monkeypatch):
    import flare_metadata as catalog

    data = tmp_path / "data"
    data.mkdir()
    path = data / "flare_position_catalog_hek.csv"
    path.write_text("fl_goescls,event_starttime,event_peaktime,event_endtime,hpc_x,hpc_y\n"
                    "X5.1,2025-11-11T00:45:00Z,2025-11-11T01:00:00Z,2025-11-11T01:10:00Z,25,-56\n",
                    encoding="utf-8")
    monkeypatch.setattr(catalog, "__file__", str(tmp_path / "flare_metadata.py"))
    assert catalog.flare_metadata(event)["peak"].startswith("2025-11-11T01:00")
    path.write_text("fl_goescls,event_starttime,event_peaktime,event_endtime,hpc_x,hpc_y\n"
                    "X5.5,2025-11-11T00:45:00Z,2025-11-11T01:00:00Z,2025-11-11T01:10:00Z,25,-56\n",
                    encoding="utf-8")
    assert catalog.flare_metadata(event)["peak"] == ""
    event.rename(event.with_name("2025-11-11_X5.1"))
    renamed = event.with_name("2025-11-11_X5.1")
    assert catalog.flare_metadata(renamed)["peak"] == ""
    path.write_text("fl_goescls,event_starttime,event_peaktime,event_endtime,hpc_x,hpc_y\n"
                    "X5.2,2025-11-11T00:45:00Z,2025-11-11T01:00:00Z,2025-11-11T01:10:00Z,25,-56\n",
                    encoding="utf-8")
    assert catalog.flare_metadata(renamed)["peak"].startswith("2025-11-11T01:00")


def test_map_times_are_exact_dataset_keys_per_product(event):
    path = event / "maps" / "map_roti.h5"
    second = "2025-11-11 01:10:00.000000"
    with h5py.File(path, "a") as file:
        file.create_dataset(f"data/{second}", data=np.array(
            [(52.0, 31.0, 0.9)], dtype=[("lat", "f4"), ("lon", "f4"), ("vals", "f4")]
        ))
    assert map_epochs(event) == {"map:roti": ["2025-11-11 01:00:00.000000", second]}
    page = render_plot_editor_page(event.parent.parent, event).decode()
    assert '"map:roti": ["2025-11-11 01:00:00.000000", "2025-11-11 01:10:00.000000"]' in page
    assert 'type="datetime-local"' not in page
    stamp, points = _map_points(path, second)
    assert stamp == second and points["vals"][0] == pytest.approx(.9)
    request = {"email": "user@example.org", "panels": [{"series": "map:roti", "epoch": second}]}
    assert validate_request(event, request)["epoch"] == second
    request["panels"][0]["epoch"] = "2025-11-11 01:05:00.000000"
    with pytest.raises(ValueError, match="available"):
        validate_request(event, request)
    request["panels"][0]["epoch"] = "2025-11-11T01:10:00"
    with pytest.raises(ValueError, match="available"):
        validate_request(event, request)


def test_one_observation_time_is_used_by_every_map_and_time_series(event, monkeypatch):
    from datetime import datetime
    from matplotlib.figure import Figure
    import flare_metadata as catalog

    first = "2025-11-11 01:00:00.000000"
    second = "2025-11-11 01:10:00.000000"
    with h5py.File(event / "maps" / "map_roti.h5", "a") as file:
        file.create_dataset(f"data/{second}", data=np.array(
            [(52.0, 31.0, .9)], dtype=[("lat", "f4"), ("lon", "f4"), ("vals", "f4")]))
    with h5py.File(event / "maps" / "map_dtec_2_10.h5", "w") as file:
        for stamp in (first, second):
            file.create_dataset(f"data/{stamp}", data=np.array(
                [(52.0, 31.0, .2)], dtype=[("lat", "f4"), ("lon", "f4"), ("vals", "f4")]))
    (event / "soho_sem.csv").write_text(
        "time,flux_01_50\n2025-11-11T01:00:00Z,100\n2025-11-11T01:10:00Z,120\n", encoding="utf-8")
    metadata = {"class": "X5.2", "date": "2025-11-11", "start": "2025-11-11T01:00:00+00:00",
                "peak": "2025-11-11T01:05:00+00:00", "end": "2025-11-11T01:10:00+00:00",
                "x": None, "y": None}
    monkeypatch.setattr(catalog, "flare_metadata", lambda _: metadata)
    panels = [{"series": "goes", "rect": {"x": .04, "y": .04, "w": .44, "h": .27}},
              {"series": "soho", "rect": {"x": .52, "y": .04, "w": .44, "h": .27}},
              {"series": "map:roti", "rect": {"x": .04, "y": .36, "w": .44, "h": .27}},
              {"series": "map:dtec_2_10", "rect": {"x": .52, "y": .36, "w": .44, "h": .27}}]
    request = {"epoch": second, "email": "person@example.org", "layout": "free", "panels": panels}
    assert validate_request(event, request)["epoch"] == second
    original = Figure.savefig

    def inspect(fig, *args, **kwargs):
        titles = [text.get_text() for text in fig.texts]
        assert sum("Observation: 01:10 UTC" in title for title in titles) == 1
        assert sum("map · 01:10 UTC" in title for title in titles) == 2
        lines = [ax.lines[-1] for ax in fig.axes if ax.lines and ax.lines[-1].get_color() == "#9a6400"]
        assert len(lines) == 2
        assert all(datetime.fromisoformat(str(line.get_xdata()[0])).hour == 1 and
                   datetime.fromisoformat(str(line.get_xdata()[0])).minute == 10 for line in lines)
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    assert render_plot(event, request).startswith(b"\x89PNG")
    panels[1]["epoch"] = first
    with pytest.raises(ValueError, match="same observation time"):
        validate_request(event, request)
    panels[1].pop("epoch")
    with h5py.File(event / "maps" / "map_dtec_2_10.h5", "a") as file:
        del file[f"data/{second}"]
    with pytest.raises(ValueError, match="available"):
        validate_request(event, request)
    with h5py.File(event / "maps" / "map_dtec_2_10.h5", "a") as file:
        del file[f"data/{first}"]
    with pytest.raises(ValueError, match="no common"):
        validate_request(event, request)


def test_observation_time_marks_time_series_even_without_map_panels(event, monkeypatch):
    from matplotlib.figure import Figure
    import flare_metadata as catalog

    selected = "2025-11-11 01:00:00.000000"
    monkeypatch.setattr(catalog, "flare_metadata", lambda _: {
        "class": "X5.2", "date": "2025-11-11", "start": "2025-11-11T00:55:00+00:00",
        "peak": "2025-11-11T01:05:00+00:00", "end": "2025-11-11T01:10:00+00:00",
        "x": None, "y": None})
    (event / "soho_sem.csv").write_text(
        "time,flux_01_50\n2025-11-11T01:00:00Z,100\n2025-11-11T01:01:00Z,120\n", encoding="utf-8")
    request = {"email": "person@example.org", "layout": "free", "epoch": selected, "panels": [
        {"series": "goes", "rect": {"x": .04, "y": .04, "w": .9, "h": .38}},
        {"series": "soho", "rect": {"x": .04, "y": .52, "w": .9, "h": .38}}]}
    assert validate_request(event, request)["epoch"] == selected
    original = Figure.savefig

    def inspect(fig, *args, **kwargs):
        assert sum(ax.lines[-1].get_color() == "#9a6400" for ax in fig.axes if ax.lines) == 2
        assert "Selected time 01:00 UTC" in [text.get_text() for text in fig.legends[0].get_texts()]
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    assert render_plot(event, request).startswith(b"\x89PNG")


def test_plot_http_create_fetch_delete_and_expiry(event, tmp_path):
    handler = partial(PrettyDirectoryHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        payload = json.dumps({"event": "X/2025-11-11_X5.2", "panels": [{"series": "goes"}],
                              "email": " Person@Example.org ", "title": "My X-ray"}).encode()
        with urlopen(Request(base + "/api/plots", data=payload, headers={"Content-Type": "application/json"})) as response:
            result = json.load(response)
        assert "X5.2 solar flare" in result["caption"]
        with urlopen(base + result["url"]) as response:
            assert response.headers["Content-Type"] == "image/png"
            assert response.read(8) == b"\x89PNG\r\n\x1a\n"
        with urlopen(base + "/editor/X/2025-11-11_X5.2") as response:
            assert b'id="plotCanvas"' in response.read()
        def search(email):
            request = Request(base + "/api/plots/search", data=json.dumps({"email": email}).encode())
            with urlopen(request) as response:
                return json.load(response)["plots"]

        assert search("person@example.org")[0]["title"] == "My X-ray"
        assert search("person@example.org")[0]["caption"] == result["caption"]
        assert search("PERSON@example.org")[0]["url"] == result["url"]
        assert search("someone@example.org") == []
        with urlopen(Request(base + "/api/plots", data=payload)) as response:
            second = json.load(response)
        assert {plot["url"] for plot in search("person@example.org")} == {result["url"], second["url"]}
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/.plot-cache/" + result["url"].split("/")[-1])
        assert error.value.code == 404
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/%2eplot-cache/" + result["url"].split("/")[-1])
        assert error.value.code == 404
        with urlopen(Request(base + result["delete_url"], method="DELETE")) as response:
            assert json.load(response) == {"deleted": True}
        assert [plot["url"] for plot in search("person@example.org")] == [second["url"]]
        with pytest.raises(HTTPError) as error:
            urlopen(base + result["url"])
        assert error.value.code == 404
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/plots", data=json.dumps({"event": "../X/2025-11-11_X5.2", "panels": [{"series": "goes"}], "email": "person@example.org"}).encode()))
        assert error.value.code == 400
        with urlopen(Request(base + second["delete_url"], method="DELETE")):
            pass
        assert search("person@example.org") == []
        free_payload = json.dumps({"event": "X/2025-11-11_X5.2", "email": "person@example.org",
                                   "layout": "free", "panels": [{"series": "goes",
                                   "rect": {"x": .05, "y": .1, "w": .75, "h": .6}}]}).encode()
        with urlopen(Request(base + "/api/plots", data=free_payload)) as response:
            free_result = json.load(response)
        with urlopen(base + free_result["url"]) as response:
            assert response.headers["Content-Type"] == "image/png"
        assert search("person@example.org")[0]["url"] == free_result["url"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_cache_cleanup_removes_only_expired_png(tmp_path):
    from plot_requests import TTL_SECONDS

    old = tmp_path / "old.png"
    recent = tmp_path / "recent.png"
    old.write_bytes(b"old")
    recent.write_bytes(b"recent")
    old.with_suffix(".json").write_text('{"email": "user@example.org"}', encoding="utf-8")
    import os
    os.utime(old, (10, 10))
    os.utime(recent, (TTL_SECONDS + 100, TTL_SECONDS + 100))
    cleanup_cache(tmp_path, now=TTL_SECONDS + 50)
    assert not old.exists()
    assert not old.with_suffix(".json").exists()
    assert recent.exists()
