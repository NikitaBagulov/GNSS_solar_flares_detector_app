import argparse
import sys
import traceback
from datetime import date
from pathlib import Path

from pipeline.runner import (
    PipelineConfig,
    run_discovery_and_download,
    run_index_calculation,
    run_plotting,
    run_preprocessing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        help="Начальная дата в формате YYYY-MM-DD",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="Конечная дата в формате YYYY-MM-DD",
    )
    parser.add_argument(
        "--min_flare_class",
        type=str,
        default="X1.0",
        help="Минимальный класс вспышки (по умолчанию: X1.0)",
    )
    parser.add_argument(
        "--state_json_path",
        type=str,
        default="./data/state.json",
        help="Путь к файлу состояния JSON (относительный или абсолютный)",
    )
    parser.add_argument(
        "--data_download_path",
        type=str,
        default="./data",
        help="Путь для сохранения загруженных данных",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["discovery", "preprocessing", "index", "plotting"],
        default=["discovery", "preprocessing", "index", "plotting"],
        help="Шаги pipeline для выполнения",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    try:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date) if args.end_date else date.today()

        if start_date > end_date:
            raise ValueError("start_date должен быть раньше end_date")

    except ValueError as error:
        print(f"Ошибка в параметрах дат: {error}")
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

    return PipelineConfig(
        start_date=start_date,
        end_date=end_date,
        min_flare_class=args.min_flare_class,
        state_json_path=state_json_path,
        data_download_path=data_download_path,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    print("Параметры запуска:")
    print(f"  Начальная дата: {config.start_date}")
    print(f"  Конечная дата: {config.end_date}")
    print(f"  Минимальный класс вспышки: {config.min_flare_class}")
    print(f"  Файл состояния: {config.state_json_path}")
    print(f"  Директория данных: {config.data_download_path}")
    print(f"  Шаги: {', '.join(args.steps)}")
    print()

    try:
        if "discovery" in args.steps:
            run_discovery_and_download(config)

        if "preprocessing" in args.steps:
            run_preprocessing(config)

        if "index" in args.steps:
            run_index_calculation(config)

        if "plotting" in args.steps:
            run_plotting(config)

        print("\nЗавершено успешно!")
    except Exception as error:
        print(f"\nПроизошла ошибка: {error}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
