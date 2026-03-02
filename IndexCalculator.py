import h5py
import numpy as np
from pathlib import Path
import datetime
import csv
from index_functions.day_night_index import compute_day_night_index
from index_functions.gsflai import compute_gsflai_index
from index_functions.isfai import compute_isfai_index
from dateutil import tz


def compute_index(dates, time_key, index_func):
    try:
        return index_func(dates, time_key)
    except Exception as e:
        print(f"Ошибка при вычислении индекса: {e}")
        return np.nan


TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


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
        # Нет угла возвышения — оставляем TEC как есть.
        return out

    elev = arr[:, 3]
    valid_elev = np.isfinite(elev) & (elev > 0.0) & (elev <= 90.0)
    if not np.any(valid_elev):
        return out

    mapping = _mapping_function(elev[valid_elev])
    out[valid_elev, 2] = arr[valid_elev, 2] / mapping
    return out


def retrieve_data(file) -> dict[datetime.datetime, np.ndarray]:
    """Загружает данные из HDF5 и возвращает словарь {datetime: NDArray}."""
    f_in = h5py.File(file, 'r')
    data = {}
    times = list(f_in['data'])[:]
    for str_time in times:
        time = datetime.datetime.strptime(str_time, TIME_FORMAT).replace(tzinfo=tz.gettz('UTC'))
        if time.second != 0:
            continue
        data[time] = f_in['data'][str_time][:]
    return data


class IndexRegistry:
    def __init__(self):
        self.index_functions = {}

    def register(self, name: str, func):
        self.index_functions[name] = func

    def compute_all(self, dates, time_key=None):
        results = {}
        for name, func in self.index_functions.items():
            results[name] = func(dates, time_key)
        return results


class IndexCalculator:
    def __init__(self, base_folder="preprocessed_maps"):
        self.base_folder = Path(base_folder)
        self.registry = IndexRegistry()
        self.registry.register("day_night_index", lambda dates, t: compute_index(dates, t, compute_day_night_index))
        self.registry.register("gsflai_index", lambda dates, t: compute_index(dates, t, compute_gsflai_index))
        self.registry.register("isfai_index", lambda dates, t: compute_index(dates, t, compute_isfai_index))
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

    def process_single_flare(self, flare_key: str, tracker=None):
        folder_path = self.base_folder / flare_key
        products = self.detect_products(folder_path)
        if not products:
            print(f"Нет файлов данных для вспышки {flare_key}")
            return

        indices_for_date = {}

        for product_type in products:
            print(f"\nОбработка продукта: {product_type}")
            file_path = folder_path / f"map_{product_type}.h5"
            output_file = folder_path / f"indices_{product_type}.csv"

            if output_file.exists():
                print(f"Индексы для {product_type} уже существуют, пропускаем вычисление.")
                indices_for_date[product_type] = output_file
                continue

            try:
                data_dict = retrieve_data(file_path)
            except Exception as e:
                print(f"Не удалось загрузить файл {file_path}: {e}")
                continue

            all_results = []
            for time_key, array in data_dict.items():
                work_array = np.asarray(array, dtype=float)
                if product_type == "tec":
                    work_array = _convert_stec_to_vtec(work_array)

                if work_array.ndim != 2 or work_array.shape[1] < 3:
                    continue

                dates = [tuple(row) for row in work_array[:, :3]]
                indices = self.registry.compute_all(dates, time_key)
                indices["time"] = time_key
                all_results.append(indices)

            if all_results:
                fieldnames = ["time"] + list(self.registry.index_functions.keys())
                with open(output_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_results)
                print(f"Индексы сохранены в {output_file}")
                indices_for_date[product_type] = output_file
            else:
                print(f"Нет данных для сохранения для продукта {product_type}")

        if tracker is not None and indices_for_date:
            tracker.register_files_for_flare(
                flare_key,
                {"indices": indices_for_date}
            )
