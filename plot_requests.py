"""Generate short-lived, user-configured plots from an event's stored data."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace


INDEX_COLUMNS = ("day_night_index", "gsflai_index", "isfai_index")
PRODUCTS = ("roti", "dtec_2_10", "dtec_10_20", "dtec_20_60")
SERIES = {"goes": "GOES X-ray flux", "soho": "SOHO/SEM EUV flux", "sun": "Solar disk"}
PLOT_STYLES = ("plotter", "simple")
SIMPLE_COLORS = {"goes": "#111111", "soho": "#006400", "day_night_index": "#333333",
                 "gsflai_index": "#006400", "isfai_index": "#333333"}
for product in PRODUCTS:
    for column in INDEX_COLUMNS:
        SERIES[f"{product}:{column}"] = f"{column.replace('_index', '').upper()} · {product.upper().replace('DTEC', 'dTEC')}"
    SERIES[f"map:{product}"] = f"{product.upper().replace('DTEC', 'dTEC')} map"

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
        if key == "sun":
            available[key] = label
            continue
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


def event_channels(event: Path) -> dict[str, list[str]]:
    """Channel names present in the event source files for the editor preview."""
    result = {}
    for key, folder, filename, columns in (
        ("goes", "goes_xray", "goes_xray.csv", ("xrsa", "xrsb")),
        ("soho", "soho_sem", "soho_sem.csv", ("flux_26_34", "flux_01_50")),
    ):
        path = event / folder / filename
        if not path.is_file():
            path = event / filename
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                names = csv.DictReader(stream).fieldnames or []
                result[key] = [column for column in columns if column in names]
        except OSError:
            continue
    return result


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
    epochs = map_epochs(event)
    selected_maps = {p.get("series") for p in panels if isinstance(p, dict)
                     and str(p.get("series", "")).startswith("map:")}
    common = set.intersection(*(set(epochs.get(key, ())) for key in selected_maps)) if selected_maps else {
        stamp for stamps in epochs.values() for stamp in stamps}
    epoch = request.get("epoch")
    legacy_values = [p.get("epoch") for p in panels if isinstance(p, dict) and p.get("epoch") is not None]
    if (epoch is not None and not isinstance(epoch, str)) or any(not isinstance(value, str) for value in legacy_values):
        raise ValueError("Select an observation time available in every selected map file")
    legacy_epochs = set(legacy_values)
    if len(legacy_epochs) > 1 or (epoch is not None and legacy_epochs and legacy_epochs != {epoch}):
        raise ValueError("All panels must use the same observation time")
    if epoch is None and legacy_epochs:
        epoch = next(iter(legacy_epochs))
    if selected_maps and not common:
        raise ValueError("Selected maps have no common observation time")
    if epoch is None and selected_maps:
        choices = sorted(common)
        epoch = choices[len(choices) // 2]
    if epoch is not None and (not isinstance(epoch, str) or epoch not in common):
        raise ValueError("Select an observation time available in every selected map file")
    rects = []
    for panel in panels:
        if not isinstance(panel, dict) or panel.get("series") not in available:
            raise ValueError("Unknown or unavailable data series")
        color = panel.get("color", "#2878a5")
        if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            raise ValueError("Invalid plot color")
        if panel["series"].startswith("map:"):
            if not epochs.get(panel["series"]):
                raise ValueError("No map epochs available")
        elif panel.get("epoch"):
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
            if any(x < ox + ow - 1e-6 and ox < x + w - 1e-6 and y < oy + oh - 1e-6 and oy < y + h - 1e-6
                   for ox, oy, ow, oh in rects):
                raise ValueError("Panels must not overlap")
            rects.append((x, y, w, h))
    layout = request.get("layout", "vertical")
    if layout not in ("vertical", "grid", "free"):
        raise ValueError("Invalid plot layout")
    style = request.get("style", "plotter")
    if style not in PLOT_STYLES:
        raise ValueError("Invalid plot style")
    title = request.get("title", "")
    if not isinstance(title, str) or len(title) > 100:
        raise ValueError("Title must be at most 100 characters")
    email = normalize_email(request.get("email"))
    return {"panels": panels, "layout": layout, "style": style, "title": title, "email": email, "epoch": epoch}


def _read_csv_channels(path: Path, columns: tuple[str, ...]):
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("Data file is too large for an interactive plot")
    series = {column: ([], []) for column in columns}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or columns[-1] not in reader.fieldnames:
            raise ValueError(f"Missing column {columns[-1]}")
        time_column = "time" if "time" in reader.fieldnames else reader.fieldnames[0]
        for row in reader:
            try:
                stamp = datetime.fromisoformat(row[time_column].strip().replace("Z", "+00:00"))
            except (KeyError, ValueError, TypeError):
                continue
            for column in columns:
                try:
                    value = float(row[column])
                    if math.isfinite(value):
                        series[column][0].append(stamp.replace(tzinfo=None))
                        series[column][1].append(value)
                except (KeyError, ValueError, TypeError):
                    continue
    return {key: pair for key, pair in series.items() if pair[0]}


def _read_csv(path: Path, value_column: str):
    return _read_csv_channels(path, (value_column,)).get(value_column, ([], []))


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


def _flare_window(metadata: dict) -> tuple[datetime, datetime] | None:
    """A shared UTC view around an identified flare; never infer times from a folder name."""
    try:
        start = datetime.fromisoformat(metadata["start"]).replace(tzinfo=None)
        end = datetime.fromisoformat(metadata["end"]).replace(tzinfo=None)
    except (KeyError, ValueError, TypeError):
        return None
    if end < start:
        return None
    peak = datetime.fromisoformat(metadata["peak"]).replace(tzinfo=None) if metadata.get("peak") else None
    middle = peak if peak and start <= peak <= end else start + (end - start) / 2
    half = timedelta(minutes=15) if peak else max(timedelta(minutes=15), (end - start) / 2 + timedelta(minutes=5))
    return middle - half, middle + half


def _relative_flux(measurements: dict) -> dict | None:
    """Compare different spectral channels by their change from an early baseline."""
    if any(len(values) < 2 for _, values in measurements.values()):
        return None
    result = {}
    for channel, (dates, values) in measurements.items():
        baseline = statistics.median(values[:max(1, len(values) // 10)])
        if baseline <= 0:
            return None
        result[channel] = (dates, [(value / baseline - 1) * 100 for value in values])
    return result


def render_plot(event: Path, request: dict) -> bytes:
    request = validate_request(event, request)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib import dates as mdates
    from matplotlib.dates import AutoDateLocator
    from matplotlib.lines import Line2D
    import cartopy.crs as ccrs
    from Plotter import Plotter, CombinedPlotter, PLOT_STYLE, DEFAULT_PARAMS
    from flare_metadata import flare_metadata, scientific_caption

    panels = request["panels"]
    simple = request["style"] == "simple"
    palette = ({**PLOT_STYLE, "ink": "#111111", "muted": "#333333", "grid": "#e2e2e2",
                "figure": "#ffffff", "panel": "#ffffff"} if simple else PLOT_STYLE)
    params = ({**DEFAULT_PARAMS, "text.color": palette["ink"], "axes.labelcolor": palette["ink"],
               "axes.edgecolor": palette["ink"], "xtick.color": palette["ink"],
               "ytick.color": palette["ink"], "axes.titleweight": "normal",
               "savefig.facecolor": palette["figure"]} if simple else DEFAULT_PARAMS)
    metadata = flare_metadata(event)
    heading, subtitle = scientific_caption(metadata, [panel["series"] for panel in panels])
    observation_time = datetime.fromisoformat(request["epoch"]).replace(tzinfo=None) if request["epoch"] else None
    peak_time = datetime.fromisoformat(metadata["peak"]).replace(tzinfo=None) if metadata.get("peak") else None
    if observation_time and observation_time != peak_time:
        subtitle += ("  ·  " if subtitle else "") + f"Observation: {observation_time:%H:%M} UTC"
    window = _flare_window(metadata)
    if window and observation_time:
        window = (min(window[0], observation_time - timedelta(minutes=5)),
                  max(window[1], observation_time + timedelta(minutes=5)))
    free = request["layout"] == "free"
    columns = 2 if request["layout"] == "grid" and len(panels) > 1 else 1
    rows = (len(panels) + columns - 1) // columns
    with PLOT_LOCK, plt.rc_context(params):
        fig = plt.figure(figsize=(12, 8.5), facecolor=palette["figure"]) if free else None
        if not free:
            fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 3.6 * rows), squeeze=False, constrained_layout=True)
        try:
            fig.patch.set_facecolor(palette["figure"])
            relative_flux_used = False
            for index, panel in enumerate(panels):
                key = panel["series"]
                if free:
                    rect = panel["rect"]
                    left = .055 + rect["x"] * .89
                    bottom = .055 + (1 - rect["y"] - rect["h"]) * .82
                    box_w, box_h = rect["w"] * .89, rect["h"] * .82
                    map_panel = key.startswith("map:")
                    sun_panel = key == "sun"
                    time_left = .19 if rect["w"] < .55 else .14
                    ax = fig.add_axes((left + box_w * (.1 if sun_panel else .15 if map_panel else time_left),
                                         bottom + box_h * (.14 if sun_panel else .20 if map_panel else .19),
                                        box_w * (.8 if sun_panel else .73 if map_panel else .94 - time_left),
                                        box_h * (.69 if sun_panel else .62 if map_panel else .64)),
                                      projection=ccrs.PlateCarree() if map_panel else None)
                    font_size = max(7, min(10, 10 * rect["w"] / .44, 10 * rect["h"] / .27))
                    title = SERIES[key]
                    if map_panel:
                        stamp = request["epoch"]
                        if stamp:
                            title = f"Global {key[4:].upper().replace('DTEC', 'dTEC')} map · {datetime.fromisoformat(stamp):%H:%M} UTC"
                    fig.text(left + box_w * .025, bottom + box_h * .94,
                             chr(65 + index), color=palette["ink"], weight="bold", fontsize=font_size + 3,
                             va="center")
                    fig.text(left + box_w * .5, bottom + box_h * .94,
                              title, color=palette["ink"], weight="normal" if simple else "bold",
                              fontsize=font_size, va="center", ha="center", clip_on=True)
                    if not sun_panel:
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
                color = panel.get("color", (SIMPLE_COLORS if simple else original_colors).get(
                    key.split(":")[-1], "#333333" if simple else "#2878a5"))
                if key == "sun":
                    image = None
                    for path in sorted((event / "solar_image").glob("*")):
                        if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                            try:
                                image = plt.imread(path)
                                break
                            except (OSError, ValueError, SyntaxError):
                                pass
                    painter = object.__new__(Plotter)
                    painter.data = SimpleNamespace(sun_image=image)
                    location = (metadata["x"], metadata["y"]) if metadata["x"] is not None else None
                    painter._plot_sun(ax, SimpleNamespace(location=location))
                    if simple:
                        ax.set_facecolor(palette["panel"])
                    if free:
                        ax.set_title("", loc="left")
                        if location:
                            fig.text(left + box_w * .5, bottom + box_h * .035,
                                     f"HPC: ({location[0]:.0f}, {location[1]:.0f})″",
                                     color=palette["ink"], fontsize=max(7, font_size - 1), ha="center")
                    else:
                        ax.set_title("", loc="left")
                        ax.set_title("Solar disk", loc="center", color=palette["ink"])
                    if metadata["x"] is None:
                        ax.text(.5, .02, "Flare position unavailable", transform=ax.transAxes,
                                 ha="center", va="bottom", color=palette["muted"], fontsize=8,
                                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
                elif key.startswith("map:"):
                    product = key[4:]
                    stamp, points = _map_points(event / "maps" / f"map_{product}.h5", request["epoch"])
                    painter = object.__new__(Plotter)
                    painter.data = SimpleNamespace(product_values=[{product: points}], timestamps=[datetime.fromisoformat(stamp)])
                    vmin, vmax = CombinedPlotter._get_product_color_range(product)
                    painter._plot_map(ax, 0, product_name=product, map_time=datetime.fromisoformat(stamp),
                                        vmin=vmin, vmax=vmax)
                    if simple and len(ax.collections) >= 3:
                        # The large double-stroked subsolar X and 30 pt samples from
                        # Plotter obscure the geography in a print-sized figure.
                        ax.collections[0].set_sizes([6])
                        ax.collections[-2].set_visible(False)
                        ax.collections[-1].set_sizes([28])
                        ax.collections[-1].set_linewidths([1.2])
                        ax.collections[-1].set_color("#ad6115")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    if free:
                        ax.set_title("", loc="center")  # The panel title is inside the draggable box.
                    else:
                        ax.set_title(f"{SERIES[key]} · {datetime.fromisoformat(stamp):%Y-%m-%d %H:%M} UTC", loc="center")
                else:
                    if key == "goes":
                        data_columns, path = ("xrsa", "xrsb"), event / "goes_xray" / "goes_xray.csv"
                        if not path.is_file(): path = event / "goes_xray.csv"
                    elif key == "soho":
                        data_columns, path = ("flux_26_34", "flux_01_50"), event / "soho_sem" / "soho_sem.csv"
                        if not path.is_file(): path = event / "soho_sem.csv"
                    else:
                        product, column = key.split(":", 1)
                        path = event / "indices" / f"indices_{product}.csv"
                        data_columns = (column,)
                    measurements = _read_csv_channels(path, data_columns)
                    if data_columns[-1] not in measurements:
                        raise ValueError(f"No usable data for {SERIES[key]}")
                    if window:
                        for column_name, (dates, values) in list(measurements.items()):
                            visible = [(stamp, value) for stamp, value in zip(dates, values)
                                       if window[0] <= stamp <= window[1]]
                            if visible:
                                measurements[column_name] = tuple(zip(*visible))
                            elif column_name != data_columns[-1]:
                                del measurements[column_name]
                    if key == "goes" or key == "soho":
                        relative = _relative_flux(measurements) if free else None
                        if relative is not None:
                            measurements = relative
                            relative_flux_used = True
                        elif all(value > 0 for _, values in measurements.values() for value in values) and key == "goes":
                            ax.set_yscale("log")
                        ylabel = "Flux change (%)" if relative is not None else (
                            "Flux (W m⁻²)" if key == "goes" else "EUV (photons cm⁻² s⁻¹)")
                    else:
                        ylabel = {"day_night_index": "Day/night", "gsflai_index": "GSFLAI", "isfai_index": "ISFAI"}[column]
                    for column_name, (dates, values) in measurements.items():
                        secondary = column_name in ("xrsa", "flux_26_34")
                        line_color = ("#2878a5" if column_name == "xrsa" else "#333333") if secondary else color
                        if simple and len(measurements) > 1 and "color" not in panel and column_name == "xrsb":
                            line_color = "#d1495b"
                        # The two EUV bands need both color and stroke differentiation.
                        line_style = "--" if column_name == "flux_26_34" else "-"
                        channel_label = {"xrsa": "GOES XRS-A (0.05–0.4 nm)",
                                         "xrsb": "GOES XRS-B (0.1–0.8 nm)",
                                         "flux_26_34": "SOHO/SEM 26–34 nm",
                                         "flux_01_50": "SOHO/SEM 0.1–50 nm"}.get(column_name)
                        ax.plot(dates, values, color=line_color, linewidth=1.3 if simple else 2,
                                linestyle=line_style, label=channel_label if len(measurements) > 1 else None,
                                solid_capstyle="round", marker="o" if len(dates) == 1 else None)
                    if len(measurements) > 1:
                        ax.legend(loc="lower right", framealpha=.85, fontsize=max(6, font_size - 2) if free else 8)
                    if window and any(window[0] <= stamp <= window[1]
                                      for dates, _ in measurements.values() for stamp in dates):
                        ax.set_xlim(*window)
                    elif len(measurements[data_columns[-1]][0]) == 1:
                        stamp = measurements[data_columns[-1]][0][0]
                        ax.set_xlim(stamp - timedelta(minutes=10), stamp + timedelta(minutes=10))
                    if peak_time:
                        ax.axvline(peak_time, color="#bd4651", linewidth=1.4, linestyle="--", alpha=.95)
                    if observation_time and observation_time != peak_time:
                        ax.axvline(observation_time, color="#9a6400", linewidth=1.4, linestyle=":", alpha=.95)
                    if not free:
                        ax.set_title(SERIES[key], loc="center", color=palette["ink"])
                    # Put physical units in the heading for compact free panels: vertical labels
                    # were extending into the neighboring panel in publication-sized figures.
                    wide = free and rect["w"] >= .75
                    ax.set_ylabel(ylabel if not free or wide else "", color=palette["ink"],
                                  fontsize=max(7, font_size - 2) if free else None)
                    if free and not wide:
                        fig.text(left + box_w * .045, bottom + box_h * .83,
                                  ylabel, color=palette["muted"], fontsize=max(6, font_size - 2), va="center")
                    ax.set_xlabel("Time (UTC)", fontsize=max(7, font_size - 2) if free else None)
                    ax.xaxis.set_major_locator(AutoDateLocator(maxticks=(3 if rect["w"] < .48 else 5) if free else 10))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                    if (free and simple and ax.get_yscale() == "linear" and
                            (key not in ("goes", "soho") or relative is None)):
                        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
                        ax.yaxis.get_offset_text().set_fontsize(7)
                    if not simple:
                        ax.spines["top"].set_visible(False)
                    ax.grid(True, color=palette["grid"], linewidth=.65)
                    ax.set_axisbelow(True)
            if not free:
                for index in range(len(panels), rows * columns):
                    axes[index // columns][index % columns].set_visible(False)
            if relative_flux_used:
                fig.text(.5, .012, "Flux change relative to the median of the first 10% of visible samples in each channel",
                         ha="center", color=palette["muted"], fontsize=7)
            caption = "  ·  ".join(filter(None, (subtitle, request["title"])))
            fig.suptitle(heading if free else heading + ("\n" + caption if caption else ""),
                         fontsize=16 if free else 13,
                         fontweight="bold", color=palette["ink"], y=.985)
            if caption and free:
                fig.text(.5, .955, caption, ha="center", va="top",
                         fontsize=9, color=palette["muted"])
            if free and any(panel["series"] != "sun" and not panel["series"].startswith("map:") for panel in panels):
                legend = []
                if peak_time and observation_time and observation_time != peak_time:
                    legend.append(Line2D([], [], color="#bd4651", linestyle="--", label="Flare peak"))
                    legend.append(Line2D([], [], color="#9a6400", linestyle=":", linewidth=1.4,
                                         label=f"Selected time {observation_time:%H:%M} UTC"))
                if legend:
                    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .903),
                               ncol=min(len(legend), 4), frameon=False, fontsize=8,
                               labelcolor=palette["muted"])
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
