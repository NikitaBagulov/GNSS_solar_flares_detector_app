from typing import List, Optional
from pathlib import Path
from datetime import timedelta
import json
import csv

import numpy as np
import pandas as pd
import h5py
import matplotlib.image as mpimg

from PlotData import PlotData, FlareData


class PlotDataLoader:
    def __init__(self, flares_file: str, state_file: str, sun_image_path: Optional[str] = None):
        self.flares_file = Path(flares_file)
        self.state_file = Path(state_file)
        self.sun_image_path = Path(sun_image_path) if sun_image_path else None

        # CSV со вспышками
        self.flares_df = pd.read_csv(
            self.flares_file,
            parse_dates=["start_time", "peak_time", "end_time"]
        )

        # JSON с путями к данным
        with open(self.state_file, "r", encoding="utf-8") as f:
            self.state = json.load(f)
        self.sun_image = self._load_sun_image()

    # ------------------------------------------------------------------
    # Основной метод
    # ------------------------------------------------------------------

    def load_day(self, date_str: str) -> PlotData | None:
        # 1️⃣ Вспышки за день
        day_flares = self.flares_df[
            self.flares_df["start_time"].dt.strftime("%Y-%m-%d") == date_str
        ]

        if day_flares.empty:
            return None

        # 2️⃣ Общий интервал
        start_interval = day_flares["start_time"].min() - timedelta(minutes=15)
        end_interval   = day_flares["end_time"].max() + timedelta(minutes=15)
        
        start_interval = pd.to_datetime(start_interval).tz_localize('UTC')
        end_interval = pd.to_datetime(end_interval).tz_localize('UTC')

        files_for_date = self.state["files_by_date"].get(date_str, {})
        maps_paths    = files_for_date.get("maps", {})
        indices_paths = files_for_date.get("indices", {})
        xray_path     = files_for_date.get("goes_xray")
        euv_path      = files_for_date.get("soho_sem")

        # 3️⃣ Карты (HDF5)
        timestamps, product_values = self._load_maps(
            maps_paths, start_interval, end_interval
        )

        # 4️⃣ Индексы
        index_times, day_night_index = self._load_index_csv(
            indices_paths.get("roti"),
            "day_night_index",
            start_interval,
            end_interval
        )

        _, gsflai_index = self._load_index_csv(
            indices_paths.get("roti"),
            "gsflai_index",
            start_interval,
            end_interval
        )

        _, isfai_index = self._load_index_csv(
            indices_paths.get("roti"),
            "isfai_index",
            start_interval,
            end_interval
        )

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
        flare_list: List[FlareData] = []
        for idx, row in day_flares.iterrows():
            flare_list.append(
                FlareData(
                    flare_id=idx,
                    duration=(row.end_time - row.start_time).total_seconds() / 60,
                    start_time=row.start_time,
                    peak_time=row.peak_time,
                    end_time=row.end_time,
                    location=(row.hpc_x, row.hpc_y)
                )
            )

        # 7️⃣ Результат
        return PlotData(
            timestamps=timestamps,
            product_values=product_values,

            xray_times=xray_times,
            xray_values=xray_values,

            euv_times=euv_times,
            euv_values=euv_values,

            index_times=index_times,
            day_night_index=day_night_index,
            gsflai_index=gsflai_index,
            isfai_index=isfai_index,

            flare=flare_list,
            sun_image=self.sun_image
        )

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _load_maps(self, maps_paths, start_interval, end_interval):
        """
        Загружает карты из HDF5 и возвращает:
        - timestamps: общий список всех уникальных временных меток
        - product_values: список словарей для каждого времени, ключи — названия продуктов
        """
        timestamps = set()
        product_data = {}

        for prod_name, path in maps_paths.items():
            path = Path(path)
            if not path.exists():
                continue

            product_data[prod_name] = {}
            with h5py.File(path, "r") as f:
                for str_time in f["data"]:
                    time = pd.to_datetime(str_time, utc=True)
                    if start_interval <= time <= end_interval:
                        timestamps.add(time)
                        # сохраняем все точки
                        product_data[prod_name][time] = f["data"][str_time][:]  

        timestamps = sorted(timestamps)

        product_values = []
        for t in timestamps:
            prod_dict = {}
            for prod_name in maps_paths.keys():
                prod_dict[prod_name] = product_data[prod_name].get(t, np.array([]))
            product_values.append(prod_dict)

        return timestamps, product_values


    def _load_index_csv(self, path, column_name, start_interval, end_interval):
        if not path or not Path(path).exists():
            return [], []

        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        mask = (df["time"] >= start_interval) & (df["time"] <= end_interval)
        df = df[mask]

        return df["time"].tolist(), df[column_name].tolist()


    def _load_csv_interval(self, path, value_col, start_interval, end_interval):
        if not path or not Path(path).exists():
            return [], []

        df = pd.read_csv(path)
        times = pd.to_datetime(df.iloc[:, 0], utc=True)

        mask = (times >= start_interval) & (times <= end_interval)
        df = df[mask]

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
