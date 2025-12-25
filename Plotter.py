import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import AutoDateLocator
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import numpy as np

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
    def __init__(self, plot_data, products_to_plot=None):
        self.data = plot_data
        self.products_to_plot = products_to_plot or list(plot_data.product_values[0].keys())

    def plot_all(self):
        for i, map_time in enumerate(self.data.timestamps):
            fig = plt.figure(figsize=(10, 9), constrained_layout=True)
            gs = fig.add_gridspec(3, 1, height_ratios=[6, 2, 2])

            ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
            ax_indices = fig.add_subplot(gs[1])
            ax_solar = fig.add_subplot(gs[2])

            self._plot_map(ax_map, i)
            self._plot_indices(ax_indices, map_time)
            self._plot_solar(ax_solar, map_time)

            plt.show()
            plt.close(fig)

    def _plot_map(self, ax, time_index, product_name="dtec_2_10"):
        if not self.data.product_values or time_index >= len(self.data.product_values):
            ax.set_title("No map data or invalid index")
            return
        print(time_index)
        points = self.data.product_values[time_index].get(product_name, np.array([]))
        if points.size == 0:
            ax.set_title(f"No data for product {product_name}")
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
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_title(f"{product_name} Map @ {self.data.timestamps[time_index]}")
        plt.colorbar(sc, ax=ax, label=product_name)

    def _plot_indices(self, ax, highlight_time=None):
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
            ax.axvline(highlight_time, color='red', linestyle='--', label='Current Time')

        ax.set_ylabel("Normalized Index")
        ax.set_title("Indices (Normalized)")
        ax.legend()
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
            ax.axvline(highlight_time, color='red', linestyle='--', label='Current Time')

        ax.set_ylabel("Normalized Flux")
        ax.set_title("Solar Activity (Normalized)")
        ax.grid(True)
        ax.legend()
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
