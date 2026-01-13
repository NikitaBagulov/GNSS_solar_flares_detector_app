from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import AutoDateLocator
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.patches as patches
import math

DEFAULT_PARAMS = {
    'font.size': 18,
    'figure.dpi': 100,
    'font.family': 'serif',
    'font.weight': 'light',
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 12
}
plt.rcParams.update(DEFAULT_PARAMS)

MAP_POINT_SIZE = 45
DEFAULT_CMAPS = {
    "roti": "viridis",
    "dtec_2_10": "plasma",
    "dtec_10_20": "cividis",
    "dtec_20_60": "inferno"
}
MAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "custom_cmap", ["blue", "cyan", "yellow", "red"]
)

class Plotter:
    def __init__(self, plot_data, products_to_plot=None, output_dir="results"):
        self.data = plot_data
        if products_to_plot is not None:
            self.products_to_plot = products_to_plot
        elif plot_data.product_values:
            self.products_to_plot = list(plot_data.product_values[0].keys())
        else:
            self.products_to_plot = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_all(self):
        for i, map_time in enumerate(self.data.timestamps):
            fig = plt.figure(figsize=(15, 9), constrained_layout=False)
            gs = fig.add_gridspec(
                3,
                3,
                height_ratios=[6, 2, 2],
                width_ratios=[3.4, 0.15, 1.4],
                wspace=0.25,
                hspace=0.35,
            )

            ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
            ax_cbar = fig.add_subplot(gs[0, 1])
            ax_sun = fig.add_subplot(gs[0, 2])
            ax_indices = fig.add_subplot(gs[1, :])
            ax_solar = fig.add_subplot(gs[2, :])

            self._plot_map(ax_map, ax_cbar, i, map_time=map_time)
            flare = self._select_nearest_flare(map_time)
            self._plot_sun(ax_sun, flare)
            self._plot_indices(ax_indices, map_time, flare)
            self._plot_solar(ax_solar, map_time)
            self._format_time_axis(ax_indices)
            self._format_time_axis(ax_solar)

            title = f"GNSS Solar Flare Analysis — {map_time:%Y-%m-%d %H:%M UTC}"
            if flare:
                title += f"\nFlare {flare.flare_id}: {flare.start_time:%H:%M}–{flare.end_time:%H:%M}"
            fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
            fig.subplots_adjust(top=0.9, bottom=0.07)

            output_path = self._build_output_path(map_time, flare)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

    def _plot_map(self, ax, cbar_ax, time_index, product_name="dtec_2_10", map_time=None):
        if not self.data.product_values or time_index >= len(self.data.product_values):
            ax.set_title("No map data or invalid index")
            cbar_ax.set_axis_off()
            return
        points = self.data.product_values[time_index].get(product_name, np.array([]))
        if points.size == 0:
            ax.set_title(f"No data for product {product_name}")
            cbar_ax.set_axis_off()
            return

        all_lats, all_lons, all_vals = [], [], []
        for p in points:
            all_lats.append(p['lat'])
            all_lons.append(p['lon'])
            all_vals.append(p['vals'])

        sc = ax.scatter(
            all_lons, all_lats, c=all_vals,
            s=MAP_POINT_SIZE, cmap=MAP_CMAP,
            alpha=0.85,
            transform=ccrs.PlateCarree()
        )
        ax.coastlines()
        ax.gridlines(draw_labels=True, linewidth=0.2, color="gray", alpha=0.6, linestyle="--")
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        if map_time is None:
            map_time = self.data.timestamps[time_index]
        ax.set_title(f"{product_name} @ {map_time:%Y-%m-%d %H:%M UTC}", fontsize=14, pad=6)
        plt.colorbar(sc, cax=cbar_ax, label=product_name)
        if map_time:
            self._plot_terminator(ax, map_time)

    def _plot_sun(self, ax, flare):
        ax.set_title("Solar Disk")
        ax.axis("off")
        img = self.data.sun_image
        if img is not None:
            ax.imshow(img, origin="upper")
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
            if flare and flare.location and all(np.isfinite(flare.location)):
                x_arcsec, y_arcsec = flare.location
                x_pixel, y_pixel, radius_px = self._convert_hpc_to_pixel(img, x_arcsec, y_arcsec)
                if x_pixel is not None:
                    ax.scatter([x_pixel], [y_pixel], s=120, color="red", marker="*", edgecolor="white", linewidth=0.8)
                    ax.annotate(
                        "Flare",
                        (x_pixel, y_pixel),
                        xytext=(10, -10),
                        textcoords="offset points",
                        color="white",
                        fontsize=10,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
                    )
                    ax.add_patch(plt.Circle((img.shape[1] / 2, img.shape[0] / 2), radius_px, fill=False, color="white", alpha=0.3))
            return

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_facecolor("#0b1026")
        sun = patches.Circle((0.5, 0.5), 0.45, facecolor="#f9d94a", edgecolor="#f6a800", linewidth=2, alpha=0.95)
        ax.add_patch(sun)
        ax.add_patch(patches.Circle((0.5, 0.5), 0.45, fill=False, edgecolor="white", alpha=0.4, linewidth=1))

        if flare and flare.location and all(np.isfinite(flare.location)):
            x_arcsec, y_arcsec = flare.location
            x_norm, y_norm = self._convert_hpc_to_axes(x_arcsec, y_arcsec)
            if x_norm is not None:
                ax.scatter([x_norm], [y_norm], s=120, color="red", marker="*", edgecolor="white", linewidth=0.8)
                ax.annotate(
                    "Flare",
                    (x_norm, y_norm),
                    xytext=(10, -10),
                    textcoords="offset points",
                    color="white",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
                )

    def _plot_indices(self, ax, highlight_time=None, flare=None):
        times = self._to_naive(self.data.index_times)
        data_lists = [
            ("Day/Night", self.data.day_night_index, "orange"),
            ("GSFLAI", self.data.gsflai_index, "green"),
            ("ISFAI", self.data.isfai_index, "blue")
        ]

        # Нормализация в [0,1] для каждого набора
        for label, values, color in data_lists:
            if values:
                values = np.array(values)
                norm_values = (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))
                ax.plot(times, norm_values, label=label, color=color)

        if highlight_time:
            ax.axvline(
                self._ensure_naive_time(highlight_time),
                color='red',
                linestyle='--',
                linewidth=1,
                label="_nolegend_",
            )

        if flare:
            self._plot_flare_markers(ax, flare)

        ax.set_ylabel("Normalized Index")
        ax.set_title("Indices (Normalized)")
        ax.legend(ncol=3, loc="upper left", frameon=True, fontsize=10)
        ax.grid(True)
        ax.set_ylim(0, 1)  # Все линии в одной высоте

    def _plot_solar(self, ax, highlight_time=None):
        times_values = [
            (self.data.xray_times, self.data.xray_values, 'X-ray', 'purple'),
            (self.data.euv_times, self.data.euv_values, 'EUV', 'brown')
        ]

        for times, values, label, color in times_values:
            if values and times:
                values = np.array(values)
                naive_times = self._to_naive(times)
                norm_values = (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))
                ax.plot(naive_times, norm_values, label=label, color=color)

        if highlight_time:
            ax.axvline(
                self._ensure_naive_time(highlight_time),
                color='red',
                linestyle='--',
                linewidth=1,
                label="_nolegend_",
            )

        ax.set_ylabel("Normalized Flux")
        ax.set_title("Solar Activity (Normalized)")
        ax.grid(True)
        ax.legend(ncol=2, loc="upper left", frameon=True, fontsize=10)
        ax.set_ylim(0, 1)




    def _format_time_axis(self, ax):
        ax.xaxis.set_major_locator(AutoDateLocator(maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))

    def _to_naive(self, times):
        """Конвертирует массив времени в tz-naive (UTC)"""
        if times is None or len(times) == 0:
            return np.array([])
        # Если это pandas Series или список с pd.Timestamp
        return np.array([t.tz_convert(None) if hasattr(t, 'tz') and t.tz else t for t in times])

    def _ensure_naive_time(self, dt_value):
        if hasattr(dt_value, "tzinfo") and dt_value.tzinfo:
            return dt_value.tz_convert(None) if hasattr(dt_value, "tz_convert") else dt_value.replace(tzinfo=None)
        return dt_value

    def _convert_hpc_to_pixel(self, img, x_arcsec, y_arcsec, solar_radius_arcsec=960.0):
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

    def _convert_hpc_to_axes(self, x_arcsec, y_arcsec, solar_radius_arcsec=960.0):
        x_norm = x_arcsec / solar_radius_arcsec
        y_norm = y_arcsec / solar_radius_arcsec
        if abs(x_norm) > 1.1 or abs(y_norm) > 1.1:
            return None, None
        return 0.5 + x_norm * 0.45, 0.5 + y_norm * 0.45

    def _plot_flare_markers(self, ax, flare):
        if flare.start_time and flare.end_time:
            ax.axvspan(
                self._ensure_naive_time(flare.start_time),
                self._ensure_naive_time(flare.end_time),
                color="#f8c471",
                alpha=0.18,
                zorder=0,
            )
        markers = [
            ("Start", flare.start_time, "#2ecc71", "-"),
            ("Peak", flare.peak_time, "#f39c12", "--"),
            ("End", flare.end_time, "#e74c3c", "-."),
        ]
        for label, time_value, color, style in markers:
            if not time_value:
                continue
            ax.axvline(
                self._ensure_naive_time(time_value),
                color=color,
                linestyle=style,
                linewidth=1.2,
                label="_nolegend_",
            )

    def _plot_terminator(self, ax, time_value=None, color="black", alpha=0.25):
        lat, lon = self._get_latlon(time_value)
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

    def _get_latlon(self, time_value=None):
        if time_value is None:
            time_value = datetime.now(timezone.utc)
        time_value = self._to_utc_datetime(time_value)
        sub_lat, sub_lon = self._subsolar_point(time_value)
        return sub_lat, sub_lon

    def _to_utc_datetime(self, dt_value):
        if dt_value.tzinfo is None:
            return dt_value.replace(tzinfo=timezone.utc)
        return dt_value.astimezone(timezone.utc)

    def _subsolar_point(self, dt_value):
        year, month = dt_value.year, dt_value.month
        day = dt_value.day + (dt_value.hour + (dt_value.minute + dt_value.second / 60.0) / 60.0) / 24.0
        if month <= 2:
            year -= 1
            month += 12
        A = year // 100
        B = 2 - A + (A // 4)
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5

        T = (jd - 2451545.0) / 36525.0
        L0 = (280.46646 + 36000.76983 * T + 0.0003032 * T * T) % 360.0
        M = (357.52911 + 35999.05029 * T - 0.0001537 * T * T) % 360.0

        Mrad = math.radians(M)
        C = (
            (1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(Mrad)
            + (0.019993 - 0.000101 * T) * math.sin(2 * Mrad)
            + 0.000289 * math.sin(3 * Mrad)
        )
        true_long = L0 + C

        eps0 = 23.439291 - 0.0130042 * T
        eps = math.radians(eps0)
        lam = math.radians(true_long)

        dec = math.asin(math.sin(eps) * math.sin(lam))
        subsolar_lat = math.degrees(dec)

        ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
        ra_deg = (math.degrees(ra) + 360.0) % 360.0

        gmst = (
            280.46061837
            + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * T * T
            - (T * T * T) / 38710000.0
        ) % 360.0

        gha = (gmst - ra_deg) % 360.0
        subsolar_lon = (180.0 - gha)
        subsolar_lon = (subsolar_lon + 180.0) % 360.0 - 180.0

        return subsolar_lat, subsolar_lon

    def _select_nearest_flare(self, map_time):
        if not self.data.flare:
            return None
        if not isinstance(map_time, datetime):
            return self.data.flare[0]

        map_time = self._ensure_naive_time(map_time)
        closest = None
        min_delta = None
        for flare in self.data.flare:
            if not flare.peak_time:
                continue
            flare_time = self._ensure_naive_time(flare.peak_time)
            delta = abs((flare_time - map_time).total_seconds())
            if min_delta is None or delta < min_delta:
                min_delta = delta
                closest = flare
        return closest or self.data.flare[0]

    def _build_output_path(self, map_time, flare):
        date_folder = map_time.strftime("%Y-%m-%d")
        if flare and flare.peak_time:
            flare_label = f"flare_{flare.peak_time:%H%M%S}"
        elif flare and flare.start_time:
            flare_label = f"flare_{flare.start_time:%H%M%S}"
        else:
            flare_label = "flare_unknown"
        time_label = map_time.strftime("%H%M%S")
        filename = f"map_{time_label}.png"
        return self.output_dir / date_folder / flare_label / filename
