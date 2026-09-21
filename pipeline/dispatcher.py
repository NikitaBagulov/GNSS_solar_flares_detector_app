"""SQLite queue dispatcher for the staged flare processing pipeline."""

from __future__ import annotations

import logging
import threading
import time
from datetime import date
from pathlib import Path

from pipeline.queue import Job, SQLiteJobQueue
from pipeline.runner import (
    PipelineConfig,
    run_discovery,
    run_download_for_date,
    run_index_calculation_for_flare,
    run_plotting_for_flare,
    run_preprocessing_for_flares,
)

LOGGER = logging.getLogger(__name__)


class QueueDispatcher:
    def __init__(self, db_path: Path, config: PipelineConfig, max_attempts: int = 3) -> None:
        self.queue = SQLiteJobQueue(db_path)
        self.config = config
        self.max_attempts = max_attempts

    def discover(self) -> int:
        result = run_discovery(self.config)
        grouped = {}
        for flare_key in result.flare_keys:
            flare_date = self._flare_date(flare_key)
            if flare_date is not None:
                grouped.setdefault(flare_date.isoformat(), []).append(flare_key)
        for target_date, flare_keys in grouped.items():
            self.queue.enqueue("download", target_date=target_date, payload={"flare_keys": flare_keys}, max_attempts=self.max_attempts)
        return len(grouped)

    def run(
        self,
        stop_event: threading.Event,
        poll_seconds: int = 5,
        stale_timeout_seconds: int = 7200,
        discovery_interval_seconds: int = 1800,
    ) -> None:
        last_discovery = time.monotonic()
        while not stop_event.is_set():
            self.queue.recover_stale(stale_timeout_seconds)
            job = self.queue.claim()
            if job is None:
                if time.monotonic() - last_discovery >= discovery_interval_seconds:
                    try:
                        self.discover()
                        last_discovery = time.monotonic()
                    except Exception:
                        LOGGER.exception("Discovery iteration failed; queue worker will retry later")
                stop_event.wait(poll_seconds)
                continue
            try:
                self._execute(job)
                self.queue.succeed(job.id)
            except Exception as error:
                LOGGER.exception("Queue job %s failed: %s", job.id, error)
                self.queue.fail(job.id, error)

    def _execute(self, job: Job) -> None:
        if job.job_type == "download":
            if not run_download_for_date(self.config, date.fromisoformat(job.target_date or "")):
                raise RuntimeError(f"download failed for {job.target_date}")
            self.queue.enqueue(
                "preprocess",
                target_date=job.target_date,
                payload={"flare_keys": job.payload.get("flare_keys", [])},
                max_attempts=self.max_attempts,
            )
            return
        if job.job_type == "preprocess":
            flare_keys = {str(key) for key in job.payload.get("flare_keys", [])}
            result = run_preprocessing_for_flares(self.config, flare_keys)
            if flare_keys - set(result.flare_keys):
                raise RuntimeError(f"preprocessing produced no maps for {sorted(flare_keys - set(result.flare_keys))}")
            for flare_key in flare_keys:
                self.queue.enqueue("index", flare_key=flare_key, target_date=job.target_date, max_attempts=self.max_attempts)
            return
        if job.job_type == "index":
            if not job.flare_key:
                raise ValueError("index job has no flare_key")
            result = run_index_calculation_for_flare(self.config, job.flare_key)
            if job.flare_key not in result.flare_keys:
                raise RuntimeError(f"index produced no result for {job.flare_key}")
            self.queue.enqueue("plot", flare_key=job.flare_key, target_date=job.target_date, max_attempts=self.max_attempts)
            return
        if job.job_type == "plot":
            if not job.flare_key:
                raise ValueError("plot job has no flare_key")
            result = run_plotting_for_flare(self.config, job.flare_key)
            if job.flare_key not in result.plotted_flare_keys:
                raise RuntimeError(f"plot produced no result for {job.flare_key}")
            return
        raise ValueError(f"unknown queue job type: {job.job_type}")

    def _flare_date(self, flare_key: str) -> date | None:
        from pipeline.runner import _load_tracker, _flare_date_for_key

        return _flare_date_for_key(_load_tracker(self.config), flare_key)
