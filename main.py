from DataManager import DataManager
from FlareTracker import FlareTracker
from DataPreprocessor import DataPreprocessor
from IndexCalculator import IndexCalculator
from PlotDataLoader import PlotDataLoader
from Plotter import Plotter, CombinedPlotter

from datetime import date, timedelta
from download_functions.euv import download_soho_sem
from download_functions.simurg_hdf import download_simurg_hdf
from download_functions.xray import download_goes_xray
from download_functions.hek_flares import download_flares

import argparse
import os
import sys
from pathlib import Path
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--start_date', type=str, required=True, 
                   help='Начальная дата в формате YYYY-MM-DD')
parser.add_argument('--end_date', type=str, default=None,
                   help='Конечная дата в формате YYYY-MM-DD')
parser.add_argument('--min_flare_class', type=str, default='X1.0',
                   help='Минимальный класс вспышки (по умолчанию: X1.0)')
parser.add_argument('--state_json_path', type=str, default='flare_tracker_state.json',
                   help='Путь к файлу состояния JSON (относительный или абсолютный)')
parser.add_argument('--data_download_path', type=str, default='./data',
                   help='Путь для сохранения загруженных данных')

args = parser.parse_args()

try:
    start_date = date.fromisoformat(args.start_date)
    
    if args.end_date:
        end_date = date.fromisoformat(args.end_date)
    else:
        end_date = date.today()

    if start_date > end_date:
        print("Ошибка: start_date должен быть раньше end_date")
        sys.exit(1)
        
except ValueError as e:
    print(f"Ошибка в формате даты: {e}")
    print("Используйте формат YYYY-MM-DD")
    sys.exit(1)

state_json_path = Path(args.state_json_path)
if not state_json_path.is_absolute():
    state_json_path = Path.cwd() / state_json_path

data_download_path = Path(args.data_download_path)
if not data_download_path.is_absolute():
    data_download_path = Path.cwd() / data_download_path

data_download_path.mkdir(parents=True, exist_ok=True)
state_json_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Параметры запуска:")
print(f"  Начальная дата: {start_date}")
print(f"  Конечная дата: {end_date}")
print(f"  Минимальный класс вспышки: {args.min_flare_class}")
print(f"  Файл состояния: {state_json_path}")
print(f"  Директория данных: {data_download_path}")
print()

try:
    data_manager = DataManager(base_download_dir=str(data_download_path))

    data_manager.register_download_function("soho_sem", download_func=download_soho_sem)
    data_manager.register_download_function("goes_xray", download_func=download_goes_xray)
    # data_manager.register_download_function("hek_flares", download_func=download_flares)
    data_manager.register_download_function("simurg_hdf", download_func=download_simurg_hdf, default_extension='.h5')
    
    print(f"Зарегистрировано функций загрузки: {len(data_manager.download_functions)}")

    for func_name in ["soho_sem", "goes_xray", "hek_flares", "simurg_hdf"]:
        if func_name in data_manager.download_functions:
            print(f"✓ Функция '{func_name}' зарегистрирована")
        else:
            print(f"✗ Функция '{func_name}' НЕ зарегистрирована")
    
    print("\nИнициализация FlareTracker...")
    tracker = FlareTracker(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        min_flare_class=args.min_flare_class,
        state_file_path=str(state_json_path)
    )
    
    print("Начинается проверка пропущенных данных...")

    print(f"Диапазон дат для обработки: с {tracker.start_date} по {tracker.end_date}")

    if hasattr(tracker, 'download_missed_data'):
        result = tracker.download_missed_data()
        print(f"Результат загрузки: {result}")
    else:
        print("Ошибка: у tracker нет метода download_missed_data")
    preprocessor = DataPreprocessor(input_root=str(data_download_path))
    processed_files = preprocessor.process_all(tracker)

    calculator = IndexCalculator()

    flare_keys = list(tracker.state.get("files_by_flare", {}).keys())

    if not flare_keys:
        print("Нет данных по вспышкам для обработки индексов.")
    else:
        for flare_key in flare_keys:
            print(f"\n=== Индексы для вспышки {flare_key} ===")
            calculator.process_single_flare(flare_key, tracker=tracker)

    print("\nЗавершено успешно!")
    loader = PlotDataLoader(tracker.all_flares_file, tracker.state_file)
    for flare_key in flare_keys:
        plot_data = loader.load_flare(flare_key)
        if not plot_data:
            print(f"Нет данных для вспышки {flare_key}, пропуск.")
            continue

        print(f"Найдено {len(plot_data.timestamps)} таймстемпов")
        print(f"Количество вспышек в наборе: {len(plot_data.flare)}")

        if plot_data.product_values:
            print("Продуктовые значения для первого таймстемпа:", plot_data.product_values[0])
        if plot_data.xray_values:
            print("X-ray значение:", plot_data.xray_values[0])
        if plot_data.euv_values:
            print("EUV значение:", plot_data.euv_values[0])

        first_flare = plot_data.flare[0]
        print("Вспышка:")
        print(f"  ID: {first_flare.flare_id}")
        print(f"  Начало: {first_flare.start_time}")
        print(f"  Пик: {first_flare.peak_time}")
        print(f"  Конец: {first_flare.end_time}")
        print(f"  Локация: {first_flare.location}")

        plotter = Plotter(plot_data, products_to_plot=["roti", "dtec_2_10", "dtec_10_20", "dtec_20_60"])
        # plotter.plot_all()
        combined_plotter = CombinedPlotter(plot_data, products_to_plot=["roti", "dtec_2_10", "dtec_10_20", "dtec_20_60"])
        combined_plotter.plot_all()
    
except Exception as e:
    print(f"\nПроизошла ошибка: {e}")
    traceback.print_exc()
    sys.exit(1)
