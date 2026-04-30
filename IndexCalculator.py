import h5py
import numpy as np
from pathlib import Path
import datetime
import csv
import glob
from index_functions.day_night_index import compute_day_night_index
from index_functions.gsflai import compute_gsflai_index
from index_functions.isfai import compute_isfai_index
from dateutil import tz
from map_filters import maybe_filter_roti_points
from results_layout import event_results_dir, legacy_event_results_dir, product_file_name

def compute_index(dates, time_key, index_func):
    try:
        return index_func(dates, time_key)
    except Exception as e:
        print(f"Ошибка при вычислении индекса: {e}")
        return np.nan

TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def retrieve_data(file) -> dict[datetime.datetime, np.ndarray]:
    """
    Загружает данные из HDF5 и возвращает словарь {datetime: NDArray}.
    """
    file_path = Path(file)
    f_in = h5py.File(file_path, 'r')
    data = {}
    times = list(f_in['data'])[:]
    for str_time in times:
        time = datetime.datetime.strptime(str_time, TIME_FORMAT).replace(tzinfo=tz.gettz('UTC'))
        if time.second != 0:
            continue
        # time = time.replace(microsecond=0)
        data[time] = maybe_filter_roti_points(f_in['data'][str_time][:], file_path.stem)
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
    def __init__(self, base_folder="results", existing_data_policy: str = "validate"):
        self.base_folder = Path(base_folder)
        self.existing_data_policy = existing_data_policy
        self.registry = IndexRegistry()
        self.registry.register("day_night_index", lambda dates, t: compute_index(dates, t, compute_day_night_index))
        self.registry.register("gsflai_index",   lambda dates, t: compute_index(dates, t, compute_gsflai_index))
        self.registry.register("isfai_index",    lambda dates, t: compute_index(dates, t, compute_isfai_index))
        self.available_products = []

    def _is_index_file_valid(self, file_path: Path) -> bool:
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                required = {"time", "day_night_index", "gsflai_index", "isfai_index"}
                return bool(rows) and required.issubset(set(reader.fieldnames or []))
        except Exception:
            return False

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

    def _map_paths_for_flare(self, flare_key: str, tracker=None) -> dict[str, Path]:
        if tracker is not None:
            maps = tracker.get_files_for_flare(flare_key).get("maps") or {}
            existing = {product: Path(path) for product, path in maps.items() if Path(path).exists()}
            if existing:
                return existing

        legacy_folder = self.base_folder / flare_key
        if legacy_folder.exists():
            return {
                path.stem.replace("map_", ""): path
                for path in legacy_folder.glob("map_*.h5")
            }

        event_maps_dirs = [
            event_results_dir(flare_key, root=self.base_folder) / "maps",
            legacy_event_results_dir(flare_key, root=self.base_folder) / "maps",
        ]
        products = ["roti", "dtec_2_10", "dtec_10_20", "dtec_20_60"]
        found = {}
        for event_maps_dir in event_maps_dirs:
            for product in products:
                if product in found:
                    continue
                matches = sorted(event_maps_dir.glob(f"map_{product}.h5"))
                if not matches:
                    matches = sorted(event_maps_dir.glob(f"map_{product}_*.h5"))
                if matches:
                    found[product] = matches[0]
        return found

    def _indices_dir_for_maps(self, flare_key: str, map_paths: dict[str, Path]) -> Path:
        if map_paths:
            first_path = next(iter(map_paths.values()))
            if first_path.parent.name == "maps":
                return first_path.parent.parent / "indices"
        return event_results_dir(flare_key, root=self.base_folder) / "indices"

    def _flare_class_for_key(self, flare_key: str, tracker=None) -> str | None:
        if tracker is None:
            return None
        flares = tracker._load_all_flares()
        if flares.empty or "flare_key" not in flares.columns or "class" not in flares.columns:
            return None
        matched = flares[flares["flare_key"] == flare_key]
        if matched.empty:
            return None
        return str(matched.iloc[0]["class"])

    def process_all_folders(self):
        flare_keys = self.scan_all_flares()

        if not flare_keys:
            print("Нет папок с данными вспышек.")
            return

        print(f"Найдено вспышек для расчёта индексов: {len(flare_keys)}")

        for flare_key in flare_keys:
            self.process_single_flare(flare_key)

    def process_single_flare(self, flare_key: str, tracker=None):
        map_paths = self._map_paths_for_flare(flare_key, tracker=tracker)
        products = list(map_paths.keys())
        if not products:
            print(f"Нет файлов данных для вспышки {flare_key}")
            return

        indices_for_date = {}  # собираем все пути к индексам
        indices_dir = self._indices_dir_for_maps(flare_key, map_paths)
        indices_dir.mkdir(parents=True, exist_ok=True)
        flare_class = self._flare_class_for_key(flare_key, tracker=tracker)

        for product_type in products:
            file_path = map_paths[product_type]
            output_file = indices_dir / product_file_name(
                "indices",
                product_type,
                flare_key,
                ".csv",
                flare_class=flare_class,
            )

            if output_file.exists():
                if self.existing_data_policy == "skip":
                    print(f"Индексы для {product_type} уже существуют, пропускаем вычисление (skip).")
                    indices_for_date[product_type] = output_file
                    continue

                if self.existing_data_policy == "overwrite":
                    output_file.unlink(missing_ok=True)
                    print(f"Индексы для {product_type} будут пересозданы (overwrite).")

                if self.existing_data_policy == "validate":
                    if self._is_index_file_valid(output_file):
                        print(f"Индексы для {product_type} валидны, пропускаем вычисление (validate).")
                        indices_for_date[product_type] = output_file
                        continue
                    print(f"Индексы для {product_type} невалидны, пересчитываем (validate).")
                    output_file.unlink(missing_ok=True)

            try:
                data_dict = retrieve_data(file_path)
            except Exception as e:
                print(f"Не удалось загрузить файл {file_path}: {e}")
                continue

            all_results = []
            for time_key, array in data_dict.items():
                dates = [tuple(row) for row in array]
                indices = self.registry.compute_all(dates, time_key)
                indices["time"] = time_key
                all_results.append(indices)

            if all_results:
                fieldnames = ["time"] + list(self.registry.index_functions.keys())
                with open(output_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_results)
                indices_for_date[product_type] = output_file
            else:
                print(f"Нет данных для сохранения для продукта {product_type}")

        if tracker is not None and indices_for_date:
            tracker.set_files_for_flare_section(flare_key, "indices", indices_for_date)
        if indices_for_date:
            print(f"Индексы для {flare_key}: сохранено продуктов {len(indices_for_date)}.")






# # ---------------- Пример использования ---------------- #
# if __name__ == "__main__":
#     calculator = IndexCalculator()
#     date = datetime.date(2012, 1, 27)

#     products = calculator.detect_products(date)
#     print(f"Найденные продукты: {products}")

#     calculator.process_folder(date)
