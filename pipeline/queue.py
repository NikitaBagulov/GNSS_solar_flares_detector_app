"""Small SQLite-backed durable job queue for the pipeline."""

from __future__ import annotations

import json
import sqlite3
import socket
import threading
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


JOB_STATUSES = ("pending", "running", "succeeded", "failed", "dead")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_flare_key(value: object) -> str:
    """Return the canonical key used for deduplication and job locking."""
    key = str(value).strip()
    if not key:
        raise ValueError("flare_key must not be empty")
    return key


@dataclass(frozen=True)
class Job:
    id: int
    job_type: str
    flare_key: str | None
    target_date: str | None
    payload: dict
    status: str
    attempts: int
    max_attempts: int
    last_error: str | None


class SQLiteJobQueue:
    def __init__(self, path: Path, worker_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.worker_id = worker_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        self._local = threading.local()
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=10000")
            connection.execute("PRAGMA foreign_keys=ON")
            self._local.connection = connection
        return connection

    def _initialize(self) -> None:
        self._connection().executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY,
                job_type TEXT NOT NULL,
                flare_key TEXT,
                target_date TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed','dead')),
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                available_at TEXT NOT NULL,
                locked_at TEXT,
                locked_by TEXT,
                started_at TEXT,
                finished_at TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(job_type, flare_key, target_date)
            );
            CREATE INDEX IF NOT EXISTS jobs_claim_idx
                ON jobs(status, available_at, id);
            """
        )

    def enqueue(
        self,
        job_type: str,
        *,
        flare_key: str | None = None,
        target_date: str | None = None,
        payload: dict | None = None,
        max_attempts: int = 3,
    ) -> int:
        if not job_type:
            raise ValueError("job_type must not be empty")
        if flare_key is not None:
            flare_key = stable_flare_key(flare_key)
        elif target_date is not None:
            # SQLite treats NULL values as distinct in UNIQUE constraints.
            # Use a deterministic synthetic key for date-scoped jobs.
            flare_key = f"@date:{target_date}"
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        now = _now()
        connection = self._connection()
        existing = connection.execute(
            "SELECT id, status FROM jobs WHERE job_type=? AND flare_key IS ? AND target_date IS ?",
            (job_type, flare_key, target_date),
        ).fetchone()
        if existing is not None:
            if existing["status"] in {"failed", "dead"}:
                connection.execute(
                    """UPDATE jobs SET status='pending', available_at=?, updated_at=?,
                       max_attempts=?, payload_json=?, last_error=NULL WHERE id=?""",
                    (now, now, max_attempts, json.dumps(payload or {}, sort_keys=True), existing["id"]),
                )
            return int(existing["id"])
        cursor = connection.execute(
            """
            INSERT INTO jobs (
                job_type, flare_key, target_date, payload_json, status,
                max_attempts, available_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?)
            ON CONFLICT(job_type, flare_key, target_date) DO UPDATE SET
                payload_json=excluded.payload_json,
                max_attempts=excluded.max_attempts,
                updated_at=excluded.updated_at
            WHERE jobs.status IN ('pending', 'failed')
            """,
            (job_type, flare_key, target_date, json.dumps(payload or {}, sort_keys=True), max_attempts, now, now, now),
        )
        if cursor.rowcount and cursor.lastrowid:
            return int(cursor.lastrowid)
        row = connection.execute(
            "SELECT id FROM jobs WHERE job_type=? AND flare_key IS ? AND target_date IS ?",
            (job_type, flare_key, target_date),
        ).fetchone()
        assert row is not None
        return int(row["id"])

    def claim(self) -> Job | None:
        connection = self._connection()
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE status='pending' AND available_at <= ?
                ORDER BY id LIMIT 1
                """,
                (_now(),),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            now = _now()
            connection.execute(
                """
                UPDATE jobs SET status='running', attempts=attempts+1,
                    locked_at=?, locked_by=?, started_at=COALESCE(started_at, ?), updated_at=?
                WHERE id=? AND status='pending'
                """,
                (now, self.worker_id, now, now, row["id"]),
            )
            connection.execute("COMMIT")
            return self._get(int(row["id"]))
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def succeed(self, job_id: int) -> None:
        self._connection().execute(
            "UPDATE jobs SET status='succeeded', finished_at=?, updated_at=?, locked_at=NULL, locked_by=NULL WHERE id=? AND status='running'",
            (_now(), _now(), job_id),
        )

    def fail(self, job_id: int, error: BaseException, retry_delay_seconds: int = 60) -> None:
        connection = self._connection()
        row = connection.execute("SELECT attempts, max_attempts FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return
        dead = row["attempts"] >= row["max_attempts"]
        status = "dead" if dead else "failed"
        available = _now() if dead else datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + retry_delay_seconds, timezone.utc
        ).isoformat()
        message = "".join(traceback.format_exception(type(error), error, error.__traceback__))[-8000:]
        connection.execute(
            """UPDATE jobs SET status=?, available_at=?, finished_at=?, updated_at=?,
               locked_at=NULL, locked_by=NULL, last_error=? WHERE id=?""",
            (status, available, _now(), _now(), message, job_id),
        )

    def recover_stale(self, timeout_seconds: int) -> int:
        cutoff = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() - timeout_seconds, timezone.utc
        ).isoformat()
        cursor = self._connection().execute(
            """UPDATE jobs SET status=CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'failed' END,
               available_at=?, locked_at=NULL, locked_by=NULL, updated_at=?
               WHERE status='running' AND locked_at < ?""",
            (_now(), _now(), cutoff),
        )
        return cursor.rowcount

    def _get(self, job_id: int) -> Job:
        row = self._connection().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        assert row is not None
        return Job(
            id=row["id"], job_type=row["job_type"], flare_key=row["flare_key"],
            target_date=row["target_date"], payload=json.loads(row["payload_json"]),
            status=row["status"], attempts=row["attempts"], max_attempts=row["max_attempts"],
            last_error=row["last_error"],
        )
