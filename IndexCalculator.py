import csv
import datetime
from pathlib import Path

import h5py
import numpy as np
from dateutil import tz

from index_functions.day_night_index import compute_day_night_index
from index_functions.gsflai import compute_gsflai_index
from index_functions.isfai import compute_isfai_index

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
DAY_NIGHT_PRODUCTS = ["roti", "dtec_2_10", "dtec_10_20", "dtec_20_60"]
TEC_PRODUCT = "tec"


def compute_index(dates, time_key, index_func):
    try:
        return index_func(dates, time_key)
    except Exception as e:
        print(f"Ошибка при вычислении индекса: {e}")
        return np.nan


def _mapping_function(elev_deg: np.ndarray, shell_height_km: float = 350.0) -> np.ndarray:
    """Single-layer mapping function M(E): STEC = M(E) * VTEC."""
    re_km = 6371.0
    elev_rad = np.radians(np.clip(elev_deg, 0.1, 90.0))
    cos_e = np.cos(elev_rad)
    ratio = re_km / (re_km + shell_height_km)
    denom = np.sqrt(np.maximum(1.0 - (ratio * cos_e) ** 2, 1e-8))
    return 1.0 / denom


def _convert_stec_to_vtec(array: np.ndarray) -> np.ndarray:
    """Converts map rows [lat, lon, tec(, elevation)] to [lat, lon, vtec]."""
    arr = np.asarray(array, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return arr

    out = arr.copy()
    if arr.shape[1] < 4:
        return out

    elev = arr[:, 3]
    valid_elev = np.isfinite(elev) & (elev > 0.0) & (elev <= 90.0)
    if not np.any(valid_elev):
        return out

    mapping = _mapping_function(elev[valid_elev])
    out[valid_elev, 2] = arr[valid_elev, 2] / mapping
    return out


def retrieve_data(file) -> dict[datetime.datetime, np.ndarray]:
    f_in = h5py.File(file, 'r')
    data = {}
    times = list(f_in['data'])[:]
    for str_time in times:
        time = datetime.datetime.strptime(str_time, TIME_FORMAT).replace(tzinfo=tz.gettz('UTC'))
        if time.second != 0:
            continue
        data[time] = f_in['data'][str_time][:]
    return data


class IndexCalculator:
    def __init__(self, base_folder="preprocessed_maps"):
        self.base_folder = Path(base_folder)
        self.available_products = []

    def scan_all_flares(self):
        flare_keys = []
        for folder in self.base_folder.iterdir():
            if folder.is_dir():
                flare_keys.append(folder.name)
        return sorted(flare_keys)

    def detect_products(self, folder_path: Path):
        if not folder_path.exists():
            return []
        product_files = folder_path.glob("map_*.h5")
        products = [f.stem.replace("map_", "") for f in product_files]
        self.available_products = products
        return products

    def process_all_folders(self):
        flare_keys = self.scan_all_flares()
        if not flare_keys:
            print("Нет папок с данными вспышек.")
            return

        print(f"Найдено {len(flare_keys)} вспышек: {flare_keys}")
        for flare_key in flare_keys:
            print(f"\n=== Обработка вспышки {flare_key} ===")
            self.process_single_flare(flare_key)

    @staticmethod
    def _prepare_rows(array: np.ndarray, product_type: str) -> list[tuple]:
        work_array = np.asarray(array, dtype=float)
        if work_array.ndim != 2 or work_array.shape[1] < 3:
            return []

        if product_type == TEC_PRODUCT:
            work_array = _convert_stec_to_vtec(work_array)

        return [tuple(row) for row in work_array[:, :3]]

    def _build_indices_for_product(self, product_type: str, data_dict: dict) -> list[dict]:
        all_results = []
        for time_key, array in data_dict.items():
            rows = self._prepare_rows(array, product_type)
            if not rows:
                continue

            result = {"time": time_key}
            if product_type in DAY_NIGHT_PRODUCTS:
                result["day_night_index"] = compute_index(rows, time_key, compute_day_night_index)
            elif product_type == TEC_PRODUCT:
                result["gsflai_index"] = compute_index(rows, time_key, compute_gsflai_index)
                result["isfai_index"] = compute_index(rows, time_key, compute_isfai_index)
            else:
                continue

            all_results.append(result)
        return all_results

    def process_single_flare(self, flare_key: str, tracker=None):
        folder_path = self.base_folder / flare_key
        products = self.detect_products(folder_path)
        if not products:
            print(f"Нет файлов данных для вспышки {flare_key}")
            return

        indices_for_flare = {
            "day_night": {},
            "flare_activity": {},
        }

        for product_type in products:
            if product_type in DAY_NIGHT_PRODUCTS:
                output_file = folder_path / f"indices_day_night_{product_type}.csv"
                fieldnames = ["time", "day_night_index"]
            elif product_type == TEC_PRODUCT:
                output_file = folder_path / "indices_flare_activity_tec.csv"
                fieldnames = ["time", "gsflai_index", "isfai_index"]
            else:
                continue

            file_path = folder_path / f"map_{product_type}.h5"
            print(f"\nОбработка продукта: {product_type}")

            if output_file.exists():
                print(f"Индексы для {product_type} уже существуют, пропускаем вычисление.")
            else:
                try:
                    data_dict = retrieve_data(file_path)
                except Exception as e:
                    print(f"Не удалось загрузить файл {file_path}: {e}")
                    continue

                all_results = self._build_indices_for_product(product_type, data_dict)
                if not all_results:
                    print(f"Нет данных для сохранения для продукта {product_type}")
                    continue

                with open(output_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_results)
                print(f"Индексы сохранены в {output_file}")

            if product_type in DAY_NIGHT_PRODUCTS:
                indices_for_flare["day_night"][product_type] = output_file
            elif product_type == TEC_PRODUCT:
                indices_for_flare["flare_activity"][product_type] = output_file

        if tracker is not None and (indices_for_flare["day_night"] or indices_for_flare["flare_activity"]):
            tracker.register_files_for_flare(flare_key, {"indices": indices_for_flare})
