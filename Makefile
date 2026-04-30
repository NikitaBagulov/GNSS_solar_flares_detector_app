PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest

START_DATE ?= 2008-01-01
END_DATE ?= 2025-12-31
MIN_FLARE_CLASS ?= X1.0
MODE ?= once
POLL_INTERVAL_SECONDS ?= 3600
DATA_DOWNLOAD_PATH ?= ./data
STATE_JSON_PATH ?= ./data/state.json
STEPS ?= discovery preprocessing index plotting
RESULTS_DIR ?= ./results
RESULTS_PORT ?= 8000

.PHONY: help install test test-verbose run run-service serve-results cli-help lint clean

help:
	@echo "Доступные цели:"
	@echo "  make install        - установить зависимости"
	@echo "  make test           - запустить тесты"
	@echo "  make test-verbose   - запустить тесты подробно"
	@echo "  make cli-help       - показать help CLI"
	@echo "  make run            - одноразовый запуск pipeline"
	@echo "  make run-service    - запуск pipeline в service-режиме"
	@echo "  make serve-results  - directory listing для папки results"
	@echo "  make clean          - удалить Python cache"

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTEST)

test-verbose:
	$(PYTEST) -v

cli-help:
	$(PYTHON) main.py --help

run:
	$(PYTHON) main.py \
		--start_date $(START_DATE) \
		--end_date $(END_DATE) \
		--min_flare_class $(MIN_FLARE_CLASS) \
		--mode once \
		--data_download_path $(DATA_DOWNLOAD_PATH) \
		--state_json_path $(STATE_JSON_PATH) \
		--steps $(STEPS)

run-service:
	$(PYTHON) main.py \
		--start_date $(START_DATE) \
		--end_date $(END_DATE) \
		--min_flare_class $(MIN_FLARE_CLASS) \
		--mode $(MODE) \
		--poll-interval-seconds $(POLL_INTERVAL_SECONDS) \
		--data_download_path $(DATA_DOWNLOAD_PATH) \
		--state_json_path $(STATE_JSON_PATH) \
		--steps $(STEPS)

serve-results:
	$(PYTHON) results_server.py --port $(RESULTS_PORT) --directory $(RESULTS_DIR)

lint:
	@echo "Lint target не настроен: в репозитории не найден конфиг линтера."

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
