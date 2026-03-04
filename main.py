from DataManager import DataManager
from FlareTracker import FlareTracker
from DataPreprocessor import DataPreprocessor
from IndexCalculator import IndexCalculator
from PlotDataLoader import PlotDataLoader
from Plotter import CombinedPlotter

from datetime import date
from download_functions.euv import download_soho_sem
from download_functions.simurg_hdf import download_simurg_hdf
from download_functions.xray import download_goes_xray

import argparse
import sys
from pathlib import Path
import traceback
import time


def parse_args():
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
    parser.add_argument('--continuous', action='store_true',
                        help='Запускать конвейер постоянно с паузой между циклами')
    parser.add_argument('--poll_interval_sec', type=int, default=600,
                        help='Интервал между циклами в секундах для continuous режима')
    return parser.parse_args()


def init_runtime(args):
    try:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date) if args.end_date else date.today()
        if start_date > end_date:
            raise ValueError("start_date должен быть раньше end_date")
    except ValueError as e:
        print(f"Ошибка в параметрах даты: {e}")
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

    print("Параметры запуска:")
    print(f"  Начальная дата: {start_date}")
    print(f"  Конечная дата: {end_date}")
    print(f"  Минимальный класс вспышки: {args.min_flare_class}")
    print(f"  Файл состояния: {state_json_path}")
    print(f"  Директория данных: {data_download_path}")
    print(f"  Непрерывный режим: {args.continuous}")
    print()

    data_manager = DataManager(base_download_dir=str(data_download_path))
    data_manager.register_download_function("soho_sem", download_func=download_soho_sem)
    data_manager.register_download_function("goes_xray", download_func=download_goes_xray)
    data_manager.register_download_function("simurg_hdf", download_func=download_simurg_hdf, default_extension='.h5')

    tracker = FlareTracker(
        data_manager=data_manager,
        start_date=start_date,
        end_date=end_date,
        min_flare_class=args.min_flare_class,
        state_file_path=str(state_json_path)
    )
    return tracker, data_download_path


def run_pipeline_cycle(tracker, data_download_path):
    flare_keys = []

    try:
        print("Начинается проверка пропущенных данных...")
        print(f"Диапазон дат для обработки: с {tracker.start_date} по {tracker.end_date}")
        result = tracker.download_missed_data(parallel_downloads=True, max_workers=4)
        print(f"Результат загрузки: {result}")
    except Exception as e:
        print(f"❌ Ошибка в модуле загрузки: {e}")
        traceback.print_exc()

    try:
        preprocessor = DataPreprocessor(input_root=str(data_download_path))
        processed = preprocessor.process_all(tracker)
        print(f"Результат препроцессинга: {processed}")
    except Exception as e:
        print(f"❌ Ошибка в модуле препроцессинга: {e}")
        traceback.print_exc()

    try:
        calculator = IndexCalculator()
        flare_keys = list(tracker.state.get("files_by_flare", {}).keys())
        if not flare_keys:
            print("Нет данных по вспышкам для обработки индексов.")
        else:
            for flare_key in flare_keys:
                print(f"\n=== Индексы для вспышки {flare_key} ===")
                try:
                    calculator.process_single_flare(flare_key, tracker=tracker)
                except Exception as flare_error:
                    print(f"Ошибка расчета индексов для {flare_key}: {flare_error}")
    except Exception as e:
        print(f"❌ Ошибка в модуле расчета индексов: {e}")
        traceback.print_exc()

    try:
        loader = PlotDataLoader(tracker.all_flares_file, tracker.state_file)
        for flare_key in flare_keys:
            plot_data = loader.load_flare(flare_key)
            if not plot_data:
                continue
            combined_plotter = CombinedPlotter(plot_data, products_to_plot=["roti", "dtec_2_10", "dtec_10_20", "dtec_20_60", "tec"])
            combined_plotter.plot_all()
    except Exception as e:
        print(f"❌ Ошибка в модуле визуализации: {e}")
        traceback.print_exc()


def main():
    args = parse_args()
    tracker, data_download_path = init_runtime(args)

    if args.continuous:
        cycle = 0
        while True:
            cycle += 1
            print(f"\n{'=' * 24} ЦИКЛ {cycle} {'=' * 24}")
            run_pipeline_cycle(tracker, data_download_path)
            print(f"Пауза {args.poll_interval_sec} секунд до следующего цикла...")
            time.sleep(args.poll_interval_sec)
    else:
        run_pipeline_cycle(tracker, data_download_path)


if __name__ == '__main__':
    main()
