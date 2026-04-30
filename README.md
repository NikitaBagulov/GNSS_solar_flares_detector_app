# GNSS Solar Flares Detector App

## Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## Запуск pipeline через Makefile

Основной запуск:

```bash
make run START_DATE=2024-11-01 END_DATE=2025-11-12 MIN_FLARE_CLASS=X5.1
```

По умолчанию pipeline выполняет шаги:

```text
discovery preprocessing index plotting
```

После `discovery` обработка идёт по одной вспышке полностью:

```text
preprocessing -> index -> plotting
```

## Полезные параметры

```bash
make help
```

Выбор шагов:

```bash
make run START_DATE=2024-11-01 END_DATE=2025-11-12 MIN_FLARE_CLASS=X5.1 STEPS="discovery preprocessing index"
```

Политика существующих данных:

```bash
make run START_DATE=2024-11-01 END_DATE=2025-11-12 EXISTING_DATA_POLICY=skip
make run START_DATE=2024-11-01 END_DATE=2025-11-12 EXISTING_DATA_POLICY=overwrite
make run START_DATE=2024-11-01 END_DATE=2025-11-12 EXISTING_DATA_POLICY=validate
```

Service-режим:

```bash
make run-service START_DATE=2024-11-01 END_DATE=2025-11-12 MIN_FLARE_CLASS=X5.1 MODE=service POLL_INTERVAL_SECONDS=3600
```

## Results directory listing

Запуск красивого просмотра папки `results`:

```bash
make serve-results
```

Открыть в браузере:

```text
http://localhost:8000
```

Если порт занят:

```bash
make serve-results RESULTS_PORT=8001
```

## Где лежат результаты

```text
results/
  X/
    2025-11-11_X5.2/
      goes_xray/
      soho_sem/
      maps/
      indices/
      graphs/
```

State и скачанные данные по умолчанию:

```text
data/state.json
data/
```

## Тесты

```bash
make test
```

или:

```bash
pytest -q
```
