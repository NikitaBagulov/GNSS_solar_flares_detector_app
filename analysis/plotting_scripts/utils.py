import re
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from results_server import scan_events
from .config import (
    PRODUCTS, PRODUCT_LABELS, PRODUCT_VMIN_VMAX, PRODUCT_CMAPS,
    FLARE_CLASSES, FLARE_CLASS_MARKERS, FLARE_CLASS_COLORS,
    TIME_WINDOW_MINUTES, SOLAR_RADIUS_ARCSEC,
    PLOT_DPI, PLOT_FIGSIZE_SINGLE, PLOT_FIGSIZE_DASHBOARD,
    MAP_POINT_SIZE, MAP_ALPHA, OUTPUT_SUBDIRS, DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS_DIR, DEFAULT_FLARES_CSV,
    GOES_XRAY_COLUMNS, SOHO_SEM_COLUMNS,
    LINE_WIDTH, LINE_WIDTH_THICK, LINE_WIDTH_THIN,
    GRID_ALPHA, GRID_LINESTYLE,
    FILL_NEGATIVE_COLOR, FILL_POSITIVE_COLOR,
    COLORS_CONTRAST, INDEX_COLORS, XRAY_COLORS, EUV_COLOR,
    FONT_SIZE, LABEL_FONT_SIZE, TICK_FONT_SIZE, LEGEND_FONT_SIZE,
)

logger = logging.getLogger(__name__)
SIMURG_MAP_TIME_KEY_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
_UTC = pd.Timestamp.utcnow().tz

TIME_FORMAT = "%H:%M UTC"


def event_file_path(results_dir: Path, event: dict, *parts: str) -> Path:
    return results_dir / event["path"] / Path(*parts)


def load_flare_catalog(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for column in ("start_time", "peak_time", "end_time"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")
    for column in ("hpc_x", "hpc_y", "class_value", "duration_min", "peak_flux"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")
    if "class" in df.columns:
        df["class"] = df["class"].astype("string").str.upper()
    return df


def load_events(results_dir: Path) -> list[dict]:
    return scan_events(results_dir)


def find_event_by_name(events: list[dict], flare_key: str) -> dict | None:
    for event in events:
        if event.get("name") == flare_key:
            return event
    return None


def normalize_time_column(df: pd.DataFrame, preferred: str = "time") -> pd.DataFrame:
    df = df.copy()
    if preferred not in df.columns:
        df = df.rename(columns={df.columns[0]: preferred})
    df[preferred] = pd.to_datetime(df[preferred], utc=True, errors="coerce")
    return df.dropna(subset=[preferred]).sort_values(preferred)


def load_hdf5_map(event: dict, results_dir: Path, product: str, time_window: Tuple[pd.Timestamp, pd.Timestamp]) -> Tuple[List[pd.Timestamp], Dict[pd.Timestamp, np.ndarray]]:
    path = event_file_path(results_dir, event, "maps", f"map_{product}.h5")
    if not path.exists():
        logger.warning(f"Map file not found: {path}")
        return [], {}

    timestamps = []
    product_data = {}

    try:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                logger.warning(f"No 'data' group in {path}")
                return [], {}
            data_group = f["data"]
            for str_time in data_group.keys():
                try:
                    time = pd.Timestamp(str_time, tz='UTC')
                except Exception:
                    continue

                if time_window[0] <= time <= time_window[1]:
                    try:
                        values = data_group[str_time][:]
                        if product == "roti":
                            values = filter_roti_points(values)
                    except Exception as exc:
                        logger.warning(f"Failed to read dataset {str_time}: {exc}")
                        continue
                    timestamps.append(time)
                    product_data[time] = values
    except Exception as exc:
        logger.warning(f"Failed to open HDF5 map {path}: {exc}")
        return [], {}

    timestamps = sorted(timestamps)
    return timestamps, product_data


def filter_roti_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    if points.dtype.names is None:
        return points
    mask = (points['vals'] >= 0) & (points['vals'] <= 5)
    return points[mask]


def load_indices_csv(event: dict, results_dir: Path, product: str, time_window: Tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    path = event_file_path(results_dir, event, "indices", f"indices_{product}.csv")
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = normalize_time_column(df, preferred="time")
    mask = (df["time"] >= time_window[0]) & (df["time"] <= time_window[1])
    return df[mask].reset_index(drop=True)


def load_goes_xray(event: dict, results_dir: Path, time_window: Tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    path = event_file_path(results_dir, event, "goes_xray", "goes_xray.csv")
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = normalize_time_column(df, preferred="time")
    for col in GOES_XRAY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = (df["time"] >= time_window[0]) & (df["time"] <= time_window[1])
    return df[mask].reset_index(drop=True)


def load_soho_sem(event: dict, results_dir: Path, time_window: Tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    path = event_file_path(results_dir, event, "soho_sem", "soho_sem.csv")
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = normalize_time_column(df, preferred="time")
    for col in SOHO_SEM_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = (df["time"] >= time_window[0]) & (df["time"] <= time_window[1])
    return df[mask].reset_index(drop=True)


def load_solar_image(event: dict, results_dir: Path) -> Optional[np.ndarray]:
    path = event_file_path(results_dir, event, "soho_eit.png")
    if path.exists():
        try:
            return mpimg.imread(path)
        except Exception:
            pass
    path = event_file_path(results_dir, event, "soho_aia.png")
    if path.exists():
        try:
            return mpimg.imread(path)
        except Exception:
            pass
    return None


def find_nearest_map_time(timestamps: List[pd.Timestamp], target_time: pd.Timestamp, tolerance_minutes: float = 5.0) -> Optional[pd.Timestamp]:
    if not timestamps:
        return None
    target = target_time.tz_convert(None) if target_time.tz else target_time
    nearest = min(timestamps, key=lambda t: abs((t.tz_convert(None) if t.tz else t) - target).total_seconds())
    delta = abs((nearest.tz_convert(None) if nearest.tz else nearest) - target).total_seconds() / 60.0
    if delta <= tolerance_minutes:
        return nearest
    return None


def get_flare_time_window(flare_peak: pd.Timestamp, window_minutes: float = TIME_WINDOW_MINUTES) -> Tuple[pd.Timestamp, pd.Timestamp]:
    half_window = pd.Timedelta(minutes=window_minutes)
    return flare_peak - half_window, flare_peak + half_window


def parse_flare_class(event: dict) -> Tuple[str | None, float]:
    candidates = [
        str(event.get("name", "")),
        str(event.get("flare_class", "")),
        str(event.get("class", "")),
    ]
    for value in candidates:
        match = re.search(r"([ABCMX])[_\s-]?(\d+(?:[._]\d+)?)", value.upper())
        if not match:
            continue
        letter = match.group(1)
        magnitude = float(match.group(2).replace("_", "."))
        return f"{letter}{magnitude:g}", letter
    return None, None


def flare_class_letter(value: object) -> str | None:
    if pd.isna(value):
        return None
    match = re.match(r"\s*([ABCMX])", str(value).upper())
    return match.group(1) if match else None


def find_flare_row(event: dict, catalog: pd.DataFrame) -> pd.Series | None:
    """Find matching flare row in catalog for an event."""
    if "flare_key" in catalog.columns:
        match = catalog[catalog["flare_key"].astype(str) == event.get("name", "")]
        if len(match) == 1:
            return match.iloc[0]

    event_name = event.get("name", "")
    if "_" in event_name:
        parts = event_name.split("_")
        if len(parts) >= 2:
            date_str = parts[0]
            class_str = parts[1]
            match = catalog[(catalog["date"] == date_str) & (catalog["class"] == class_str)]
            if len(match) == 1:
                return match.iloc[0]

    return None


def add_flare_markers(ax: plt.Axes, start_time: pd.Timestamp, peak_time: pd.Timestamp, end_time: pd.Timestamp, alpha: float = 0.15, peak_lw: float = 2.5, show_label: bool = True, show_vspan: bool = True) -> None:
    if show_vspan:
        ax.axvspan(start_time, end_time, color="orange", alpha=0.12, zorder=0)
    ax.axvline(peak_time, color="red", linestyle="--", linewidth=peak_lw, alpha=0.9, zorder=5)
    if show_label:
        ax.annotate(
            "Peak",
            xy=(peak_time, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(8, -12),
            textcoords="offset points",
            fontsize=LEGEND_FONT_SIZE,
            color="black",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, edgecolor="none"),
        )


def format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator(maxticks=8))
    ax.set_xlabel("Time (UTC)", fontsize=LABEL_FONT_SIZE)


def apply_grid(ax: plt.Axes) -> None:
    ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)


def fill_negative(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = FILL_NEGATIVE_COLOR, alpha: float = 0.3) -> None:
    ax.fill_between(x, y, 0, where=(y < 0), color=color, alpha=alpha)


def fill_positive(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = FILL_POSITIVE_COLOR, alpha: float = 0.3) -> None:
    ax.fill_between(x, y, 0, where=(y > 0), color=color, alpha=alpha)


def save_figure(fig: plt.Figure, flare_key: str, subdir: str, filename: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    out_dir = output_dir / subdir / flare_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_solar_disk_base(
    ax: plt.Axes,
    flare_hpc_x: Optional[float] = None,
    flare_hpc_y: Optional[float] = None,
    solar_image: Optional[np.ndarray] = None,
    title: str = "",
) -> None:
    if title:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.axis("off")

    if solar_image is not None:
        ax.imshow(solar_image, origin="upper")
        ax.set_xlim(0, solar_image.shape[1])
        ax.set_ylim(solar_image.shape[0], 0)
        if flare_hpc_x is not None and flare_hpc_y is not None:
            x_pixel, y_pixel, radius_px = convert_hpc_to_pixel(solar_image, flare_hpc_x, flare_hpc_y)
            if x_pixel is not None:
                ax.scatter([x_pixel], [y_pixel], s=250, color="red", marker="*",
                          edgecolor="white", linewidth=2, zorder=10)
                ax.annotate("Flare", (x_pixel, y_pixel), xytext=(12, -12),
                           textcoords="offset points", color="white", fontsize=12,
                           fontweight="bold",
                           path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
                ax.add_patch(plt.Circle((solar_image.shape[1] / 2, solar_image.shape[0] / 2),
                                        radius_px, fill=False, color="white", alpha=0.3))
        return

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_facecolor("#0b1026")
    sun = plt.Circle((0.5, 0.5), 0.45, facecolor="#f9d94a", edgecolor="#f6a800", linewidth=2, alpha=0.95)
    ax.add_patch(sun)
    ax.add_patch(plt.Circle((0.5, 0.5), 0.45, fill=False, edgecolor="white", alpha=0.4, linewidth=1))

    if flare_hpc_x is not None and flare_hpc_y is not None:
        x_norm, y_norm = convert_hpc_to_axes(flare_hpc_x, flare_hpc_y)
        if x_norm is not None:
            ax.scatter([x_norm], [y_norm], s=250, color="red", marker="*",
                      edgecolor="white", linewidth=2, zorder=10)
            ax.annotate("Flare", (x_norm, y_norm), xytext=(12, -12),
                       textcoords="offset points", color="white", fontsize=12,
                       fontweight="bold",
                       path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])


def convert_hpc_to_pixel(img: np.ndarray, x_arcsec: float, y_arcsec: float, solar_radius_arcsec: float = SOLAR_RADIUS_ARCSEC) -> Tuple[Optional[float], Optional[float], float]:
    height, width = img.shape[0], img.shape[1]
    radius_px = min(height, width) * 0.48
    center_x = width / 2
    center_y = height / 2
    x_norm = x_arcsec / solar_radius_arcsec
    y_norm = y_arcsec / solar_radius_arcsec
    if abs(x_norm) > 1.1 or abs(y_norm) > 1.1:
        return None, None, radius_px
    x_pixel = center_x + x_norm * radius_px
    y_pixel = center_y - y_norm * radius_px
    return x_pixel, y_pixel, radius_px


def convert_hpc_to_axes(x_arcsec: float, y_arcsec: float, solar_radius_arcsec: float = SOLAR_RADIUS_ARCSEC) -> Tuple[Optional[float], Optional[float]]:
    x_norm = x_arcsec / solar_radius_arcsec
    y_norm = y_arcsec / solar_radius_arcsec
    if abs(x_norm) > 1.1 or abs(y_norm) > 1.1:
        return None, None
    return 0.5 + x_norm * 0.45, 0.5 + y_norm * 0.45


def plot_global_map(
    ax: plt.Axes,
    points: np.ndarray,
    product: str,
    map_time: pd.Timestamp,
    vmin: float = None,
    vmax: float = None,
    show_colorbar: bool = True,
) -> None:
    if points.size == 0 or points.dtype.names is None:
        ax.set_title(f"No data", fontsize=TITLE_FONT_SIZE)
        return

    lats = points['lat']
    lons = points['lon']
    vals = points['vals']

    # Sort so highest values plot on top
    order = np.argsort(vals)
    lats, lons, vals = lats[order], lons[order], vals[order]

    if vmin is None or vmax is None:
        vmin, vmax = PRODUCT_VMIN_VMAX.get(product, (np.nanmin(vals), np.nanmax(vals)))

    cmap = PRODUCT_CMAPS.get(product, "viridis")

    sc = ax.scatter(
        lons, lats, c=vals,
        s=MAP_POINT_SIZE, cmap=cmap,
        vmin=vmin, vmax=vmax,
        alpha=MAP_ALPHA,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )

    ax.coastlines(linewidth=0.5, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="black")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    if show_colorbar:
        cbar_ax = ax.inset_axes([1.02, 0.1, 0.04, 0.8])
        cbar = plt.colorbar(sc, cax=cbar_ax, label=PRODUCT_LABELS.get(product, product))
        return cbar

    plot_terminator(ax, map_time)
    plot_subsolar_point(ax, map_time)


def plot_terminator(ax: plt.Axes, time_value: pd.Timestamp, color: str = "black", alpha: float = 0.25) -> None:
    lat, lon = subsolar_point(time_value)
    pole_lng = lon
    if lat > 0:
        pole_lat = -90 + lat
        central_rot_lng = 180
    else:
        pole_lat = 90 + lat
        central_rot_lng = 0

    rotated_pole = ccrs.RotatedPole(
        pole_latitude=pole_lat,
        pole_longitude=pole_lng,
        central_rotated_longitude=central_rot_lng,
    )

    x = [-90] * 181 + [90] * 181 + [-90]
    y = list(range(-90, 91)) + list(range(90, -91, -1)) + [-90]
    ax.fill(x, y, transform=rotated_pole, color=color, alpha=alpha, zorder=3)


def plot_subsolar_point(ax: plt.Axes, map_time: pd.Timestamp) -> None:
    sub_lat, sub_lon = subsolar_point(map_time)
    ax.scatter([sub_lon], [sub_lat], s=150, color="black", linewidths=10.0, marker="x", zorder=5, transform=ccrs.PlateCarree())
    ax.scatter([sub_lon], [sub_lat], s=110, color="#ffd54f", linewidths=3.8, marker="x", zorder=6, transform=ccrs.PlateCarree())


def subsolar_point(dt: pd.Timestamp) -> Tuple[float, float]:
    dt_utc = dt.tz_convert('UTC') if dt.tz else dt.tz_localize('UTC')
    year = dt_utc.year
    month = dt_utc.month
    day = dt_utc.day + (dt_utc.hour + (dt_utc.minute + dt_utc.second / 60.0) / 60.0) / 24.0

    if month <= 2:
        year -= 1
        month += 12

    A = year // 100
    B = 2 - A + (A // 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5

    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + 36000.76983 * T + 0.0003032 * T * T) % 360.0
    M = (357.52911 + 35999.05029 * T - 0.0001537 * T * T) % 360.0

    Mrad = np.radians(M)
    C = (
        (1.914602 - 0.004817 * T - 0.000014 * T * T) * np.sin(Mrad)
        + (0.019993 - 0.000101 * T) * np.sin(2 * Mrad)
        + 0.000289 * np.sin(3 * Mrad)
    )
    true_long = L0 + C

    eps0 = 23.439291 - 0.0130042 * T
    eps = np.radians(eps0)
    lam = np.radians(true_long)

    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    subsolar_lat = np.degrees(dec)

    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    ra_deg = (np.degrees(ra) + 360.0) % 360.0

    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T * T
        - (T * T * T) / 38710000.0
    ) % 360.0

    gha = (gmst - ra_deg) % 360.0
    subsolar_lon = -gha
    subsolar_lon = (subsolar_lon + 180.0) % 360.0 - 180.0

    return subsolar_lat, subsolar_lon
