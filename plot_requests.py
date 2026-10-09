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
from types import SimpleNamespace


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


def map_epochs(event: Path) -> dict[str, list[str]]:
    """Actual UTC dataset keys, grouped by the map product that owns them."""
    import h5py

    epochs = {}
    for product in PRODUCTS:
        path = event / "maps" / f"map_{product}.h5"
        if not path.is_file():
            continue
        try:
            with h5py.File(path, "r") as file:
                keys = [key for key in file["data"] if _valid_map_time(key)]
            if keys:
                epochs[f"map:{product}"] = sorted(keys)
        except (OSError, KeyError, TypeError):
            continue
    return epochs


def _valid_map_time(key: str) -> bool:
    try:
        datetime.fromisoformat(key)
        return True
    except ValueError:
        return False


def validate_request(event: Path, request: dict) -> dict:
    if not isinstance(request, dict):
        raise ValueError("Invalid plot request")
    available = event_series(event)
    panels = request.get("panels")
    if not isinstance(panels, list) or not 1 <= len(panels) <= 6:
        raise ValueError("Choose between 1 and 6 panels")
    epochs = map_epochs(event) if any(isinstance(p, dict) and str(p.get("series", "")).startswith("map:")
                                       for p in panels) else {}
    for panel in panels:
        if not isinstance(panel, dict) or panel.get("series") not in available:
            raise ValueError("Unknown or unavailable data series")
        color = panel.get("color", "#2878a5")
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("Invalid plot color")
        epoch = panel.get("epoch")
        if panel["series"].startswith("map:"):
            if not epochs.get(panel["series"]):
                raise ValueError("No map epochs available")
            if epoch is not None and epoch not in epochs[panel["series"]]:
                raise ValueError("Select a map time available in this file")
        elif epoch:
            raise ValueError("Map time is only available for map panels")
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
        keys = sorted(key for key in file["data"] if _valid_map_time(key))
        if not keys:
            raise ValueError("No map epochs available")
        key = epoch if epoch is not None else keys[len(keys) // 2]
        if key not in file["data"]:
            raise ValueError("Select a map time available in this file")
        dataset = file["data"][key]
        if not dataset.dtype.names or not all(column in dataset.dtype.names for column in ("lat", "lon", "vals")):
            raise ValueError("Map has unexpected columns")
        stride = max(1, (dataset.size + 19999) // 20000)
        points = dataset[::stride]
        return key, points


def render_plot(event: Path, request: dict) -> bytes:
    request = validate_request(event, request)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import dates as mdates
    from matplotlib.dates import AutoDateLocator
    from matplotlib.patches import Rectangle
    import cartopy.crs as ccrs
    from Plotter import Plotter, PLOT_STYLE, DEFAULT_PARAMS

    panels = request["panels"]
    free = request["layout"] == "free"
    columns = 2 if request["layout"] == "grid" and len(panels) > 1 else 1
    rows = (len(panels) + columns - 1) // columns
    with PLOT_LOCK, plt.rc_context(DEFAULT_PARAMS):
        fig = plt.figure(figsize=(12, 8.5), facecolor=PLOT_STYLE["figure"]) if free else None
        if not free:
            fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 3.6 * rows), squeeze=False, constrained_layout=True)
        try:
            fig.patch.set_facecolor(PLOT_STYLE["figure"])
            for index, panel in enumerate(panels):
                key = panel["series"]
                if free:
                    rect = panel["rect"]
                    left = .055 + rect["x"] * .89
                    bottom = .055 + (1 - rect["y"] - rect["h"]) * .82
                    box_w, box_h = rect["w"] * .89, rect["h"] * .82
                    fig.patches.append(Rectangle((left, bottom), box_w, box_h, transform=fig.transFigure,
                                                       facecolor=PLOT_STYLE["panel"], edgecolor=PLOT_STYLE["grid"],
                                                       linewidth=.8, zorder=-1))
                    map_panel = key.startswith("map:")
                    ax = fig.add_axes((left + box_w * .19, bottom + box_h * .24,
                                       box_w * (.61 if map_panel else .69), box_h * .53),
                                      projection=ccrs.PlateCarree() if map_panel else None)
                    font_size = max(7, min(11, 11 * rect["w"] / .44))
                    title = SERIES[key]
                    if map_panel:
                        title += f" · {panel.get('epoch') or 'file midpoint'} UTC"
                    fig.text(left + box_w * .04, bottom + box_h * .88,
                             title[:max(12, int(box_w * 65))], color=PLOT_STYLE["ink"],
                             weight="bold", fontsize=font_size, va="center")
                    ax.tick_params(labelsize=max(6, font_size - 3), length=3, width=.7)
                else:
                    ax = axes[index // columns][index % columns]
                    if key.startswith("map:"):
                        fig.delaxes(ax)
                        ax = fig.add_subplot(rows, columns, index + 1, projection=ccrs.PlateCarree())
                        axes[index // columns][index % columns] = ax
                original_colors = {"goes": PLOT_STYLE["xray"], "soho": PLOT_STYLE["euv"],
                                   "day_night_index": PLOT_STYLE["index_day"],
                                   "gsflai_index": PLOT_STYLE["index_gsflai"],
                                   "isfai_index": PLOT_STYLE["index_isfai"]}
                color = panel.get("color", original_colors.get(key.split(":")[-1], "#2878a5"))
                if key.startswith("map:"):
                    product = key[4:]
                    stamp, points = _map_points(event / "maps" / f"map_{product}.h5", panel.get("epoch"))
                    painter = object.__new__(Plotter)
                    painter.data = SimpleNamespace(product_values=[{product: points}], timestamps=[datetime.fromisoformat(stamp)])
                    painter._plot_map(ax, 0, product_name=product, map_time=datetime.fromisoformat(stamp),
                                      vmin=0 if product == "roti" else -1, vmax=1)
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    if free:
                        ax.set_title("")  # The panel title is inside the draggable box.
                    else:
                        ax.set_title(f"{Plotter._format_product_name(painter, product)} @ {stamp}", loc="left")
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
                    if key == "goes" or key == "soho":
                        if all(value > 0 for value in values):
                            ax.set_yscale("log")
                        ylabel = "Flux (W m⁻²)" if key == "goes" else "Flux (photons cm⁻² s⁻¹)"
                    else:
                        ylabel = {"day_night_index": "Day/night", "gsflai_index": "GSFLAI", "isfai_index": "ISFAI"}[column]
                    ax.plot(dates, values, color=color, linewidth=2, solid_capstyle="round",
                            marker="o" if len(dates) == 1 else None)
                    if not free:
                        ax.set_title(SERIES[key], loc="left", color=PLOT_STYLE["ink"])
                    ax.set_ylabel(ylabel, color=color)
                    ax.tick_params(axis="y", colors=color)
                    ax.spines["left"].set_color(color)
                    ax.set_xlabel("Time (UTC)")
                    ax.xaxis.set_major_locator(AutoDateLocator(maxticks=5 if free else 10))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                    ax.spines["top"].set_visible(False)
                    ax.grid(True, color=PLOT_STYLE["grid"], linewidth=.65)
                    ax.set_axisbelow(True)
            if not free:
                for index in range(len(panels), rows * columns):
                    axes[index // columns][index % columns].set_visible(False)
            fig.suptitle(request["title"] or event.name, fontsize=16, fontweight="bold",
                         color=PLOT_STYLE["ink"], y=.98)
            output = io.BytesIO()
            fig.savefig(output, format="png", dpi=180, facecolor=fig.get_facecolor())
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
