from typing import List, Optional
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
import h5py
import matplotlib.image as mpimg

from datetime import datetime
from dateutil import tz

from PlotData import PlotData, FlareData
from flare_utils import build_flare_key, get_flare_window
from map_filters import maybe_filter_roti_points

_UTC = tz.gettz('UTC')
SIMURG_MAP_TIME_KEY_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
logger = logging.getLogger(__name__)


def _parse_csv_times(values):
    try:
        return pd.to_datetime(values, utc=True, format="mixed")
    except TypeError:
        return pd.to_datetime(values, utc=True)


class PlotDataLoader:
    def __init__(self, flares_file: str, state_file: str, sun_image_path: Optional[str] = None):
        self.flares_file = Path(flares_file)
        self.state_file = Path(state_file)
        self.sun_image_path = Path(sun_image_path) if sun_image_path else None
        self.products = ["roti", "dtec_2_10", "dtec_10_20", "dtec_20_60"]

        # CSV со вспышками
        self.flares_df = pd.read_csv(
            self.flares_file,
            parse_dates=["start_time", "peak_time", "end_time"]
        )
        if "flare_key" not in self.flares_df.columns:
            self.flares_df["flare_key"] = self.flares_df.apply(
                lambda row: build_flare_key(
                    row.start_time,
                    row.peak_time,
                    row.end_time,
                    row.get("class"),
                ),
                axis=1,
            )

        # JSON с путями к данным
        with open(self.state_file, "r", encoding="utf-8") as f:
            self.state = json.load(f)
        self.sun_image = self._load_sun_image()

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------

    def load_flare(self, flare_key: str) -> PlotData | None:
        flare_row = self.flares_df[self.flares_df["flare_key"] == flare_key]

        if flare_row.empty:
            return None

        flare_row = flare_row.iloc[0]

        start_interval, end_interval = get_flare_window(
            flare_row.start_time,
            flare_row.end_time,
        )

        files_for_flare = self.state.get("files_by_flare", {}).get(flare_key, {})
        maps_paths = files_for_flare.get("maps", {})
        indices_paths = files_for_flare.get("indices", {})
        xray_path = files_for_flare.get("goes_xray")
        euv_path = files_for_flare.get("soho_sem")

        date_str = flare_row.start_time.strftime("%Y-%m-%d")
        files_for_date = self.state.get("files_by_date", {}).get(date_str, {})
        if not xray_path:
            xray_path = files_for_date.get("goes_xray")
        if not euv_path:
            euv_path = files_for_date.get("soho_sem")

        timestamps, product_values = self._load_maps(
            maps_paths, start_interval, end_interval
        )
        indices = {
        p: {"day_night_index": [], "gsflai_index": [], "isfai_index": []}
        for p in self.products
        }
        index_times = []
        index_times_by_product = {}
        for product in self.products:
            times, day_night_index = self._load_index_csv(
                indices_paths.get(product),
                "day_night_index",
                start_interval,
                end_interval
            )

            _, gsflai_index = self._load_index_csv(
                indices_paths.get(product),
                "gsflai_index",
                start_interval,
                end_interval
            )

            _, isfai_index = self._load_index_csv(
                indices_paths.get(product),
                "isfai_index",
                start_interval,
                end_interval
            )
            if times and not index_times:
                index_times = times
            index_times_by_product[product] = times
            indices[product]["day_night_index"] = day_night_index
            indices[product]["gsflai_index"] = gsflai_index
            indices[product]["isfai_index"] = isfai_index
        # 5️⃣ X-ray и EUV
        xray_times, xray_values = self._load_csv_interval(
            xray_path,
            value_col="xrsb",
            start_interval=start_interval,
            end_interval=end_interval
        )

        euv_times, euv_values = self._load_csv_interval(
            euv_path,
            value_col="flux_01_50",
            start_interval=start_interval,
            end_interval=end_interval
        )
        # 6️⃣ Список вспышек
        flare_list: List[FlareData] = [
            FlareData(
                flare_id=flare_key,
                duration=(flare_row.end_time - flare_row.start_time).total_seconds() / 60,
                start_time=flare_row.start_time,
                peak_time=flare_row.peak_time,
                end_time=flare_row.end_time,
                location=(flare_row.hpc_x, flare_row.hpc_y),
            )
        ]

        # 7️⃣ Результат
        return PlotData(
            timestamps=timestamps,
            product_values=product_values,



            xray_times=xray_times,
            xray_values=xray_values,

            euv_times=euv_times,
            euv_values=euv_values,

            index_times=index_times,
            index_times_by_product=index_times_by_product,
            indices=indices,

            flare=flare_list,
            sun_image=self.sun_image
        )

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _load_maps(self, maps_paths, start_interval, end_interval):
        """
        Загружает карты из HDF5 и возвращает:
        - timestamps: отсортированный список datetime
        - product_values: список словарей {product_name: ndarray}
        """

        timestamps: set[datetime] = set()
        product_data: dict[str, dict[datetime, np.ndarray]] = {}

        for prod_name, path in maps_paths.items():
            path = Path(path)
            if not path.exists():
                continue

            product_data[prod_name] = {}

            try:
                map_file = h5py.File(path, "r")
            except Exception as exc:
                logger.warning("Failed to open HDF5 map %s, skipping product %s: %s", path, prod_name, exc)
                continue

            try:
                data_group = map_file["data"]
                times = list(data_group)
            except Exception as exc:
                logger.warning("Failed to read HDF5 map %s, skipping product %s: %s", path, prod_name, exc)
                map_file.close()
                continue

            with map_file:
                for str_time in times:
                    try:
                        time = datetime.strptime(str_time, SIMURG_MAP_TIME_KEY_FORMAT)
                        time = time.replace(tzinfo=_UTC)
                    except ValueError:
                        # если формат вдруг не совпал — пропускаем
                        continue

                    if start_interval <= time <= end_interval:
                        try:
                            values = maybe_filter_roti_points(
                                data_group[str_time][:],
                                prod_name,
                            )
                        except Exception as exc:
                            logger.warning("Failed to read HDF5 dataset %s/%s: %s", path, str_time, exc)
                            continue
                        timestamps.add(time)
                        product_data[prod_name][time] = values

        # 🔁 fallback: если ничего не попало в интервал
        if not timestamps and maps_paths:
            for prod_name, path in maps_paths.items():
                path = Path(path)
                if not path.exists():
                    continue

                product_data.setdefault(prod_name, {})

                try:
                    map_file = h5py.File(path, "r")
                except Exception as exc:
                    logger.warning("Failed to open HDF5 map %s, skipping product %s: %s", path, prod_name, exc)
                    continue

                try:
                    data_group = map_file["data"]
                    times = list(data_group)
                except Exception as exc:
                    logger.warning("Failed to read HDF5 map %s, skipping product %s: %s", path, prod_name, exc)
                    map_file.close()
                    continue

                with map_file:
                    for str_time in times:
                        try:
                            time = datetime.strptime(str_time, SIMURG_MAP_TIME_KEY_FORMAT)
                            time = time.replace(tzinfo=_UTC)
                        except ValueError:
                            continue

                        try:
                            values = maybe_filter_roti_points(
                                data_group[str_time][:],
                                prod_name,
                            )
                        except Exception as exc:
                            logger.warning("Failed to read HDF5 dataset %s/%s: %s", path, str_time, exc)
                            continue
                        timestamps.add(time)
                        product_data[prod_name][time] = values

        timestamps = sorted(timestamps)

        product_values = []
        for t in timestamps:
            prod_dict = {}
            for prod_name in maps_paths.keys():
                prod_dict[prod_name] = product_data.get(prod_name, {}).get(
                    t, np.array([])
                )
            product_values.append(prod_dict)

        return timestamps, product_values


    def _load_index_csv(self, path, column_name, start_interval, end_interval):
        if not path or not Path(path).exists():
            return [], []

        df = pd.read_csv(path)
        df["time"] = _parse_csv_times(df["time"])
        mask = (df["time"] >= start_interval) & (df["time"] <= end_interval)
        df = df[mask]
        if df.empty:
            df = pd.read_csv(path)
            df["time"] = _parse_csv_times(df["time"])

        return df["time"].tolist(), df[column_name].tolist()


    def _load_csv_interval(self, path, value_col, start_interval, end_interval):
        if not path or not Path(path).exists():
            return [], []
        
        df = pd.read_csv(path)
        times = _parse_csv_times(df.iloc[:, 0])

        mask = (times >= start_interval) & (times <= end_interval)
        df = df[mask]

        if df.empty:
            df = pd.read_csv(path)
            times = _parse_csv_times(df.iloc[:, 0])
            return times.tolist(), df[value_col].tolist()

        return times[mask].tolist(), df[value_col].tolist()

    def _load_sun_image(self):
        path = None
        if self.sun_image_path and self.sun_image_path.exists():
            path = self.sun_image_path
        elif self.state.get("sun_image_path"):
            candidate = Path(self.state["sun_image_path"])
            if candidate.exists():
                path = candidate

        if not path:
            return None

        try:
            return mpimg.imread(path)
        except Exception:
            return None
