import argparse
import logging
import signal
import sys
import time
from datetime import date
from pathlib import Path
from typing import List

from pipeline.runner import (
    PipelineConfig,
    run_discovery_and_download,
    run_index_calculation,
    run_plotting,
    run_preprocessing,
)
from pipeline.run_config import RunConfig, as_module_set


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
    parser.add_argument(
        "--mode",
        choices=["once", "service"],
        default="once",
        help="Режим выполнения: once (один прогон) или service (периодический запуск)",
    )
    parser.add_argument(
        "--existing-data-policy",
        choices=["skip", "overwrite", "validate"],
        default="validate",
        help="Глобальная политика для существующих артефактов",
    )
    parser.add_argument(
        "--skip-modules",
        nargs="*",
        default=[],
        choices=["download", "preprocess", "index", "plot"],
        help="Модули, которые должны пропускать существующие артефакты",
    )
    parser.add_argument(
        "--overwrite-modules",
        nargs="*",
        default=[],
        choices=["download", "preprocess", "index", "plot"],
        help="Модули, которые должны перезаписывать артефакты",
    )
    parser.add_argument(
        "--validate-modules",
        nargs="*",
        default=[],
        choices=["download", "preprocess", "index", "plot"],
        help="Модули, которые должны валидировать артефакты",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=3600,
        help="Интервал в секундах между прогонами в service-режиме (по умолчанию: 3600)",
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

    if args.poll_interval_seconds <= 0:
        print("Ошибка: --poll-interval-seconds должен быть положительным целым числом")
        sys.exit(1)

    state_json_path = Path(args.state_json_path)
    if not state_json_path.is_absolute():
        state_json_path = Path.cwd() / state_json_path

    data_download_path = Path(args.data_download_path)
    if not data_download_path.is_absolute():
        data_download_path = Path.cwd() / data_download_path

    data_download_path.mkdir(parents=True, exist_ok=True)
    state_json_path.parent.mkdir(parents=True, exist_ok=True)

    run_config = RunConfig(
        existing_data_policy=args.existing_data_policy,
        skip_modules=as_module_set(args.skip_modules),
        overwrite_modules=as_module_set(args.overwrite_modules),
        validate_modules=as_module_set(args.validate_modules),
    )

    return PipelineConfig(
        start_date=start_date,
        end_date=end_date,
        min_flare_class=args.min_flare_class,
        state_json_path=state_json_path,
        data_download_path=data_download_path,
        run_config=run_config,
    )


def run_pipeline_once(config: PipelineConfig, steps: List[str]) -> None:
    if "discovery" in steps:
        run_discovery_and_download(config)

    if "preprocessing" in steps:
        run_preprocessing(config)

    if "index" in steps:
        run_index_calculation(config)

    if "plotting" in steps:
        run_plotting(config)


def run_orchestration(
    config: PipelineConfig,
    steps: List[str],
    mode: str,
    poll_interval_seconds: int,
) -> None:
    logger = logging.getLogger(__name__)
    stop_requested = False

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        signal_name = signal.Signals(signum).name
        logger.info("Получен сигнал %s. Запрошена корректная остановка.", signal_name)
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    consecutive_errors = 0

    while not stop_requested:
        try:
            run_pipeline_once(config=config, steps=steps)
            consecutive_errors = 0

            if mode == "once":
                logger.info("Pipeline завершен успешно в режиме once.")
                return

            sleep_seconds = poll_interval_seconds
            logger.info("Итерация завершена. Следующий запуск через %s сек.", sleep_seconds)
        except Exception:
            consecutive_errors += 1
            logger.exception(
                "Ошибка в итерации pipeline (ошибка #%s подряд).",
                consecutive_errors,
            )

            if mode == "once":
                raise

            sleep_seconds = min(
                poll_interval_seconds,
                5 * (2 ** (consecutive_errors - 1)),
            )
            logger.info(
                "Повторный запуск через %s сек (backoff после ошибки).",
                sleep_seconds,
            )

        elapsed = 0
        while elapsed < sleep_seconds and not stop_requested:
            chunk = min(1, sleep_seconds - elapsed)
            time.sleep(chunk)
            elapsed += chunk

    logger.info("Pipeline остановлен корректно.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()
    config = build_config(args)

    logging.info("Параметры запуска:")
    logging.info("  Начальная дата: %s", config.start_date)
    logging.info("  Конечная дата: %s", config.end_date)
    logging.info("  Минимальный класс вспышки: %s", config.min_flare_class)
    logging.info("  Файл состояния: %s", config.state_json_path)
    logging.info("  Директория данных: %s", config.data_download_path)
    logging.info("  Шаги: %s", ", ".join(args.steps))
    logging.info("  Режим: %s", args.mode)
    logging.info("  Интервал опроса (сек): %s", args.poll_interval_seconds)
    logging.info("  Глобальная policy: %s", args.existing_data_policy)
    logging.info("  Skip modules: %s", ", ".join(args.skip_modules) if args.skip_modules else "-")
    logging.info("  Overwrite modules: %s", ", ".join(args.overwrite_modules) if args.overwrite_modules else "-")
    logging.info("  Validate modules: %s", ", ".join(args.validate_modules) if args.validate_modules else "-")

    try:
        run_orchestration(
            config=config,
            steps=args.steps,
            mode=args.mode,
            poll_interval_seconds=args.poll_interval_seconds,
        )
        logging.info("Завершено успешно!")
    except Exception:
        logging.exception("Произошла ошибка.")
        sys.exit(1)


if __name__ == "__main__":
    main()
