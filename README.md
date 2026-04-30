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

## Быстрый запуск

1. Установить зависимости:

```bash
make install
```

2. Запустить обработку с параметрами по умолчанию:

```bash
make run
```

3. Открыть результаты в браузере:

```bash
make serve-results
```

Затем откройте:

```text
http://localhost:8000
```

## Запуск с параметрами

Чаще всего нужно поменять только даты и минимальный класс вспышки:

```bash
make run START_DATE=2024-11-01 END_DATE=2025-11-12 MIN_FLARE_CLASS=X5.1
```

Если нужно запустить не все шаги:

```bash
make run START_DATE=2024-11-01 END_DATE=2025-11-12 MIN_FLARE_CLASS=X5.1 STEPS="discovery preprocessing index"
```

Если нужно выбрать поведение для уже существующих данных:

```bash
make run START_DATE=2024-11-01 END_DATE=2025-11-12 EXISTING_DATA_POLICY=skip
make run START_DATE=2024-11-01 END_DATE=2025-11-12 EXISTING_DATA_POLICY=overwrite
make run START_DATE=2024-11-01 END_DATE=2025-11-12 EXISTING_DATA_POLICY=validate
```

Service-режим с повтором раз в час:

```bash
make run-service START_DATE=2024-11-01 END_DATE=2025-11-12 MIN_FLARE_CLASS=X5.1 MODE=service POLL_INTERVAL_SECONDS=3600
```

Список всех команд:

```bash
make help
```

По умолчанию pipeline выполняет шаги:

```text
discovery preprocessing index plotting
```

После поиска вспышек pipeline идёт по одной вспышке полностью:

```text
download -> preprocessing -> index -> plotting
```

## Results directory listing

Если порт `8000` занят:

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

При загрузке `simurg_hdf` в консоли должен появляться размер файла и прогресс загрузки. Ограничения по времени загрузки не задаются.

## Тесты

```bash
make test
```

или:

```bash
pytest -q
```
