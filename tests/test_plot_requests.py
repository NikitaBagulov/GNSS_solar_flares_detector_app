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

from plot_requests import cleanup_cache, render_plot, validate_request
from results_server import PrettyDirectoryHandler


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
        {"series": "goes", "color": "#2255aa", "rect": {"x": .04, "y": .04, "w": .66, "h": .3}},
        {"series": "map:roti", "rect": {"x": .3, "y": .46, "w": .62, "h": .48}},
    ]}
    with Image.open(io.BytesIO(render_plot(event, request))) as image:
        assert abs(image.width / image.height - 12 / 8.5) < .01
    request["panels"][1]["rect"]["w"] = .8
    with pytest.raises(ValueError, match="canvas"):
        validate_request(event, request)
    request["panels"][1]["rect"]["w"] = float("nan")
    with pytest.raises(ValueError, match="position"):
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
