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
    f_in = h5py.File(file, 'r')
    data = {}
    times = list(f_in['data'])[:]
    for str_time in times:
        time = datetime.datetime.strptime(str_time, TIME_FORMAT).replace(tzinfo=tz.gettz('UTC'))
        if time.second != 0:
            continue
        # time = time.replace(microsecond=0)
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
        self.registry.register("gsflai_index",   lambda dates, t: compute_index(dates, t, compute_gsflai_index))
        self.registry.register("isfai_index",    lambda dates, t: compute_index(dates, t, compute_isfai_index))
        self.available_products = []

    # -------- NEW --------
    def scan_all_dates(self):
        dates = []
        for folder in self.base_folder.iterdir():
            if folder.is_dir():
                try:
                    d = datetime.datetime.strptime(folder.name, "%Y-%m-%d").date()
                    dates.append(d)
                except ValueError:
                    pass
        return sorted(dates)

    def detect_products(self, date: datetime.date):
        print(date)
        folder_path = self.base_folder / date.strftime("%Y-%m-%d")
        print(folder_path)
        if not folder_path.exists():
            return []
        product_files = folder_path.glob("map_*.h5")
        products = [f.stem.replace("map_", "") for f in product_files]
        self.available_products = products
        return products

    # -------- UPDATED --------
    def process_all_folders(self):
        all_dates = self.scan_all_dates()

        if not all_dates:
            print("Нет папок с датами.")
            return

        print(f"Найдено {len(all_dates)} дат: {all_dates}")

        for date in all_dates:
            print(f"\n=== Обработка даты {date} ===")
            self.process_single_date(date)

    # оставляем старую функцию, но делаем её внутренней
    def process_single_date(self, date: datetime.date, tracker=None):
        products = self.detect_products(date)
        if not products:
            print(f"Нет файлов данных для даты {date}")
            return

        folder_path = self.base_folder / date.strftime("%Y-%m-%d")

        indices_for_date = {}  # собираем все пути к индексам

        for product_type in products:
            print(f"\nОбработка продукта: {product_type}")
            file_path = folder_path / f"map_{product_type}.h5"
            output_file = folder_path / f"indices_{product_type}.csv"

            # Проверка существующего CSV
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
                print(f"Индексы сохранены в {output_file}")
                indices_for_date[product_type] = output_file
            else:
                print(f"Нет данных для сохранения для продукта {product_type}")

        # Регистрируем все файлы разом
        if tracker is not None and indices_for_date:
            tracker.register_files_for_date(
                date,
                {"indices": indices_for_date}
            )






# # ---------------- Пример использования ---------------- #
# if __name__ == "__main__":
#     calculator = IndexCalculator()
#     date = datetime.date(2012, 1, 27)

#     products = calculator.detect_products(date)
#     print(f"Найденные продукты: {products}")

#     calculator.process_folder(date)
