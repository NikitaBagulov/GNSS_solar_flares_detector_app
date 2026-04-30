# GNSS Solar Flares Detector App

## Назначение сервиса и структура пайплайна

Сервис запускает конвейер обработки данных о солнечных вспышках и GNSS-наблюдениях: 

1. **discovery** — поиск вспышек в заданном диапазоне дат и скачивание исходных данных;
2. **preprocessing** — подготовка HDF-карт по каждому событию;
3. **index** — расчёт индексов по подготовленным данным;
4. **plotting** — построение графиков/изображений по картам и индексам.

Состав шагов управляется CLI-флагом `--steps` (доступные значения: `discovery preprocessing index plotting`).

---

## Требования и подготовка окружения

Минимально:

- Python 3.10+;
- доступ в интернет к внешним источникам данных;
- системные и Python-зависимости проекта (например: `pandas`, `sunpy`, `astropy`, `python-dateutil`, а также зависимости для модулей препроцессинга/индексов/визуализации).

Быстрый старт:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
# Установите зависимости проекта (если у вас есть requirements/lock-файл, используйте его)
```

Проверка CLI:

```bash
python main.py --help
```

---

## Примеры команд

> Во всех примерах подставьте актуальные даты и, при необходимости, пути `--data_download_path` / `--state_json_path`.

### 1) Одноразовый запуск

```bash
python main.py \
  --start_date 2024-01-01 \
  --end_date 2024-01-31 \
  --min_flare_class X1.0 \
  --mode once
```

```bash
python main.py --start_date 2024-11-01 --end_date 2025-11-12 --min_flare_class X5.1 --mode once
```

### 2) Service-режим (1 час)

```bash
python main.py \
  --start_date 2024-01-01 \
  --end_date 2024-01-31 \
  --mode service \
  --poll-interval-seconds 3600
```

### 3) Skip существующих данных

```bash
python main.py \
  --start_date 2024-01-01 \
  --end_date 2024-01-31 \
  --existing-data-policy skip
```

### 4) Полная перезапись

```bash
python main.py \
  --start_date 2024-01-01 \
  --end_date 2024-01-31 \
  --existing-data-policy overwrite
```

### 5) Выборочное отключение модулей

Пример: пропустить построение графиков.

```bash
python main.py \
  --start_date 2024-01-01 \
  --end_date 2024-01-31 \
  --steps discovery preprocessing index
```

### 6) Выборочная перезапись/валидация модулей

Глобально `skip`, но:
- для `download` — принудительная перезапись,
- для `index` — валидация.

```bash
python main.py \
  --start_date 2024-01-01 \
  --end_date 2024-01-31 \
  --existing-data-policy skip \
  --overwrite-modules download \
  --validate-modules index
```

Дополнительно можно точно задать `--skip-modules` / `--overwrite-modules` / `--validate-modules` для модулей: `download`, `preprocess`, `index`, `plot`.

---

## Где хранятся state и артефакты

По умолчанию:

- **State-файл**: `./data/state.json` (задаётся через `--state_json_path`);
- **Список всех вспышек**: `all_flares.csv` рядом с state-файлом;
- **Скачанные данные**: внутри `--data_download_path` (по умолчанию `./data`) в структуре `YYYY-MM-DD/<source>/...`;
- **Публичные артефакты для directory listing**: `./results/<class-letter>/<YYYY-MM-DD>_<class>/...`, например `./results/X/2025-11-11_X5.2/...`;
- **GOES X-ray**: `goes_xray/goes_xray.csv`;
- **SOHO SEM**: `soho_sem/soho_sem.csv`;
- **Карты**: `maps/map_<product>.h5`;
- **Индексы**: `indices/indices_<product>.csv`;
- **Графики**: `graphs/<product>/map_<product>_<HH-MM-SS>_UTC.png`, `graphs/combined/...`.

---

## Directory listing результатов

Обычный просмотр папки `results` через браузер:

```bash
make serve-results
```

Откройте:

```text
http://localhost:8000
```

Если порт занят:

```bash
make serve-results RESULTS_PORT=8001
```

---

## Типовые ошибки и диагностика

1. **Неверные даты (`start_date > end_date` или неверный формат)**
   - Симптом: ошибка валидации параметров дат при старте.
   - Что проверить: формат `YYYY-MM-DD`, корректный диапазон.

2. **Некорректный интервал service-опроса**
   - Симптом: ошибка `--poll-interval-seconds должен быть положительным целым числом`.
   - Что проверить: значение `> 0`.

3. **Проблемы с внешними источниками или зависимостями**
   - Симптом: исключения во время `discovery`/download шагов.
   - Что проверить:
     - сетевой доступ;
     - установку библиотек (`sunpy`, `astropy` и др.);
     - права на запись в `--data_download_path`.

4. **Конфликт ожиданий по политике существующих данных**
   - Симптом: файлы «не перезаписываются» или «перезаписываются неожиданно».
   - Что проверить:
     - глобальный `--existing-data-policy`;
     - модульные переопределения `--skip-modules`, `--overwrite-modules`, `--validate-modules`.

5. **Нет выходных графиков/индексов для части вспышек**
   - Симптом: пустые папки или отсутствие файлов для конкретного flare.
   - Что проверить:
     - какие шаги включены в `--steps`;
     - наличие валидных входов из предыдущего шага;
     - логи конкретной итерации пайплайна.
