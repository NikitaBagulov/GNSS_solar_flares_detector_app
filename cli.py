import argparse


MODULE_CHOICES = ["download", "preprocess", "index", "plot"]
POLICY_CHOICES = ["skip", "overwrite", "validate"]
MODE_CHOICES = ["once", "service"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GNSS solar flares detector pipeline runner")
    parser.add_argument("--start_date", type=str, required=True, help="Начальная дата в формате YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, default=None, help="Конечная дата в формате YYYY-MM-DD")
    parser.add_argument("--mode", choices=MODE_CHOICES, default="once", help="Режим запуска")
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=3600,
        help="Интервал опроса в сервисном режиме",
    )
    parser.add_argument(
        "--existing-data-policy",
        choices=POLICY_CHOICES,
        default="skip",
        help="Глобальная политика работы с существующими данными",
    )
    parser.add_argument(
        "--skip-modules",
        nargs="*",
        default=[],
        choices=MODULE_CHOICES,
        help="Модули, которые нужно пропустить",
    )
    parser.add_argument(
        "--overwrite-modules",
        nargs="*",
        default=[],
        choices=MODULE_CHOICES,
        help="Модули, для которых применить полную перезапись",
    )
    parser.add_argument(
        "--validate-modules",
        nargs="*",
        default=[],
        choices=MODULE_CHOICES,
        help="Модули, для которых применять проверку существующих данных",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)
