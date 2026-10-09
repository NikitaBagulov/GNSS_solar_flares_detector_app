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

from plot_requests import SERIES, SIMPLE_COLORS, _flare_window, cleanup_cache, map_epochs, render_plot, validate_request, _map_points
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


def test_publication_labels_and_focused_flare_window(event, monkeypatch):
    from datetime import datetime
    from matplotlib.figure import Figure
    import flare_metadata as catalog

    (event / "goes_xray.csv").write_text(
        "time,xrsb\n2025-11-11T01:00:00Z,0.001\n2025-11-11T12:00:00Z,0.9\n", encoding="utf-8")
    metadata = {"class": "X5.2", "date": "2025-11-11", "start": "2025-11-11T01:00:00+00:00",
                "peak": "2025-11-11T01:05:00+00:00", "end": "2025-11-11T01:10:00+00:00",
                "x": None, "y": None}
    monkeypatch.setattr(catalog, "flare_metadata", lambda _: metadata)
    assert _flare_window(metadata) == (datetime(2025, 11, 11, 0, 35), datetime(2025, 11, 11, 1, 35))
    observed = []
    original_savefig = Figure.savefig

    def inspect_figure(fig, *args, **kwargs):
        titles = [text.get_text() for text in fig.texts]
        assert any("ROTI map · 2025-11-11 01:00 UTC" == text for text in titles)
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
        assert len(ax.lines) == 4  # X-ray flux plus onset, peak and end markers.
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
    assert "X5.2" in heading and "Peak 01:05 UTC" in subtitle and "HPC (100″, -200″)" in subtitle
    (tmp_path / "data" / "all_flares.csv").write_text(
        "class,start_time,peak_time,end_time,hpc_x,hpc_y\n"
        "X5.2,2025-11-11T01:00:00Z,2025-11-11T01:05:00Z,2025-11-11T01:10:00Z,,\n", encoding="utf-8")
    assert flare_metadata(event)["x"] is None


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
    assert validate_request(event, request)["panels"][0]["epoch"] == second
    request["panels"][0]["epoch"] = "2025-11-11 01:05:00.000000"
    with pytest.raises(ValueError, match="available"):
        validate_request(event, request)
    request["panels"][0]["epoch"] = "2025-11-11T01:10:00"
    with pytest.raises(ValueError, match="available"):
        validate_request(event, request)


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
