"""Generate short-lived, user-configured plots from an event's stored data."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path


INDEX_COLUMNS = ("day_night_index", "gsflai_index", "isfai_index")
PRODUCTS = ("roti", "dtec_2_10", "dtec_10_20", "dtec_20_60")
SERIES = {"goes": "GOES X-ray", "soho": "SOHO SEM"}
for product in PRODUCTS:
    for column in INDEX_COLUMNS:
        SERIES[f"{product}:{column}"] = f"{product} · {column}"
    SERIES[f"map:{product}"] = f"Map · {product}"

PLOT_LOCK = threading.Lock()
TTL_SECONDS = 3 * 24 * 60 * 60
MAX_CSV_BYTES = 20 * 1024 * 1024
EMAIL_PATTERN = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")


def normalize_email(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Enter a valid email address")
    email = value.strip().lower()
    if len(email) > 254 or not EMAIL_PATTERN.fullmatch(email):
        raise ValueError("Enter a valid email address")
    return email


def event_series(event: Path) -> dict[str, str]:
    available = {}
    for key, label in SERIES.items():
        if key == "goes":
            path = event / "goes_xray" / "goes_xray.csv"
            path = path if path.is_file() else event / "goes_xray.csv"
        elif key == "soho":
            path = event / "soho_sem" / "soho_sem.csv"
            path = path if path.is_file() else event / "soho_sem.csv"
        elif key.startswith("map:"):
            path = event / "maps" / f"map_{key[4:]}.h5"
        else:
            path = event / "indices" / f"indices_{key.split(':', 1)[0]}.csv"
        if path.is_file():
            if ":" in key and not key.startswith("map:"):
                try:
                    with path.open(encoding="utf-8-sig", newline="") as stream:
                        if key.split(":", 1)[1] not in (csv.DictReader(stream).fieldnames or []):
                            continue
                except OSError:
                    continue
            available[key] = label
    return available


def validate_request(event: Path, request: dict) -> dict:
    if not isinstance(request, dict):
        raise ValueError("Invalid plot request")
    available = event_series(event)
    panels = request.get("panels")
    if not isinstance(panels, list) or not 1 <= len(panels) <= 6:
        raise ValueError("Choose between 1 and 6 panels")
    for panel in panels:
        if not isinstance(panel, dict) or panel.get("series") not in available:
            raise ValueError("Unknown or unavailable data series")
        color = panel.get("color", "#2878a5")
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("Invalid plot color")
        epoch = panel.get("epoch")
        if epoch:
            if not isinstance(epoch, str) or len(epoch) > 32:
                raise ValueError("Invalid map time")
            try:
                datetime.fromisoformat(epoch)
            except ValueError as exc:
                raise ValueError("Invalid map time") from exc
        if request.get("layout") == "free":
            rect = panel.get("rect")
            if not isinstance(rect, dict) or any(
                not isinstance(rect.get(axis), (int, float)) or isinstance(rect.get(axis), bool)
                or not math.isfinite(rect[axis]) for axis in ("x", "y", "w", "h")
            ):
                raise ValueError("Invalid panel position")
            x, y, w, h = (rect[axis] for axis in ("x", "y", "w", "h"))
            if x < 0 or y < 0 or w < .16 or h < .16 or x + w > 1.000001 or y + h > 1.000001:
                raise ValueError("Panel must fit inside the canvas")
    layout = request.get("layout", "vertical")
    if layout not in ("vertical", "grid", "free"):
        raise ValueError("Invalid plot layout")
    title = request.get("title", "")
    if not isinstance(title, str) or len(title) > 100:
        raise ValueError("Title must be at most 100 characters")
    email = normalize_email(request.get("email"))
    return {"panels": panels, "layout": layout, "title": title, "email": email}


def _read_csv(path: Path, value_column: str):
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("Data file is too large for an interactive plot")
    dates, values = [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or value_column not in reader.fieldnames:
            raise ValueError(f"Missing column {value_column}")
        time_column = "time" if "time" in reader.fieldnames else reader.fieldnames[0]
        for row in reader:
            try:
                stamp = datetime.fromisoformat(row[time_column].strip().replace("Z", "+00:00"))
                value = float(row[value_column])
                if value != value or abs(value) == float("inf"):
                    continue
                dates.append(stamp.replace(tzinfo=None))
                values.append(value)
            except (KeyError, ValueError, TypeError):
                continue
    return dates, values


def _map_points(path: Path, epoch: str | None):
    import h5py

    with h5py.File(path, "r") as file:
        if "data" not in file or not file["data"].keys():
            raise ValueError("No map epochs available")
        keys = sorted(file["data"].keys())
        key = min(keys, key=lambda candidate: abs((datetime.fromisoformat(candidate) - datetime.fromisoformat(epoch)).total_seconds())) if epoch else keys[len(keys) // 2]
        dataset = file["data"][key]
        if not dataset.dtype.names or not all(column in dataset.dtype.names for column in ("lat", "lon", "vals")):
            raise ValueError("Map has unexpected columns")
        stride = max(1, (dataset.size + 19999) // 20000)
        points = dataset[::stride]
        return key, points["lon"], points["lat"], points["vals"]


def render_plot(event: Path, request: dict) -> bytes:
    request = validate_request(event, request)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    panels = request["panels"]
    free = request["layout"] == "free"
    columns = 2 if request["layout"] == "grid" and len(panels) > 1 else 1
    rows = (len(panels) + columns - 1) // columns
    with PLOT_LOCK:
        fig = plt.figure(figsize=(12, 8.5)) if free else None
        if not free:
            fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 3.6 * rows), squeeze=False, constrained_layout=True)
        try:
            fig.patch.set_facecolor("#f6f7f9")
            for index, panel in enumerate(panels):
                key = panel["series"]
                if free:
                    rect = panel["rect"]
                    ax = fig.add_axes((.055 + rect["x"] * .89, .055 + (1 - rect["y"] - rect["h"]) * .82,
                                       rect["w"] * .89, rect["h"] * .82))
                    ax.set_facecolor("white")
                    ax.tick_params(labelsize=8)
                else:
                    ax = axes[index // columns][index % columns]
                color = panel.get("color", "#2878a5")
                if key.startswith("map:"):
                    product = key[4:]
                    stamp, lon, lat, vals = _map_points(event / "maps" / f"map_{product}.h5", panel.get("epoch"))
                    cmap = LinearSegmentedColormap.from_list("chosen", ["#f6f7f9", color])
                    scatter = ax.scatter(lon, lat, c=vals, s=4, cmap=cmap, rasterized=True)
                    fig.colorbar(scatter, ax=ax, shrink=.7)
                    ax.set(xlabel="Longitude", ylabel="Latitude", xlim=(-180, 180), ylim=(-90, 90))
                    ax.set_title(f"{SERIES[key]} · {stamp}")
                else:
                    if key == "goes":
                        column, path = "xrsb", event / "goes_xray" / "goes_xray.csv"
                        if not path.is_file(): path = event / "goes_xray.csv"
                    elif key == "soho":
                        column, path = "flux_01_50", event / "soho_sem" / "soho_sem.csv"
                        if not path.is_file(): path = event / "soho_sem.csv"
                    else:
                        product, column = key.split(":", 1)
                        path = event / "indices" / f"indices_{product}.csv"
                    dates, values = _read_csv(path, column)
                    if not dates:
                        raise ValueError(f"No usable data for {SERIES[key]}")
                    ax.plot(dates, values, color=color, linewidth=1.6)
                    ax.set_title(SERIES[key])
                    ax.set_xlabel("UTC")
                    ax.tick_params(axis="x", labelrotation=20)
                ax.grid(alpha=.2)
            if not free:
                for index in range(len(panels), rows * columns):
                    axes[index // columns][index % columns].set_visible(False)
            fig.suptitle(request["title"] or event.name, fontsize=15, y=.98)
            output = io.BytesIO()
            fig.savefig(output, format="png", dpi=135)
            return output.getvalue()
        finally:
            plt.close(fig)


def cleanup_cache(cache: Path, now: float | None = None) -> None:
    if not cache.is_dir():
        return
    cutoff = (now if now is not None else time.time()) - TTL_SECONDS
    for path in cache.glob("*.png"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                path.with_suffix(".json").unlink(missing_ok=True)
        except OSError:
            pass
    for path in cache.glob("*.json"):
        if not path.with_suffix(".png").is_file():
            path.unlink(missing_ok=True)


def store_plot(cache: Path, image: bytes, email: str, event: str, title: str) -> str:
    cache.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.png"
    image_path = cache / name
    with image_path.open("xb") as output:
        output.write(image)
    try:
        image_path.with_suffix(".json").write_text(
            json.dumps({"email": email, "event": event, "title": title}, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        image_path.unlink(missing_ok=True)
        raise
    return name


def plots_for_email(cache: Path, email: str) -> list[dict]:
    email = normalize_email(email)
    cleanup_cache(cache)
    plots = []
    for path in cache.glob("*.json"):
        try:
            image = path.with_suffix(".png")
            data = json.loads(path.read_text(encoding="utf-8"))
            if not image.is_file() or data.get("email") != email:
                continue
            created = image.stat().st_mtime
            plots.append({"url": f"/generated/{image.name}", "delete_url": f"/api/plots/{image.name}",
                          "event": data["event"], "title": data["title"],
                          "created_at": datetime.fromtimestamp(created).isoformat(timespec="seconds"),
                          "expires_at": datetime.fromtimestamp(created + TTL_SECONDS).isoformat(timespec="seconds")})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(plots, key=lambda item: item["created_at"], reverse=True)
