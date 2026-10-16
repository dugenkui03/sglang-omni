"""Persist workloads and observations with bounded, paginated reads."""

import json
import math
import sqlite3
import threading
import time
from pathlib import Path

from tts_workbench.config import RunConfig
from tts_workbench.records import (
    EventRecord,
    GpuSample,
    RequestPage,
    RequestRecord,
    RunRecord,
    RunStatistics,
    RunStatus,
    TimelinePoint,
)

ACTIVE_STATUSES = ("running", "stopping")


class Store:
    """Serialize short SQLite transactions; callers offload them to threads."""

    def __init__(self, root: Path) -> None:
        """Initialize only the workbench-owned result directory and database."""
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.audio_root = self.root / "audio"
        self.audio_root.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(
            self.root / "state.sqlite3", check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.statistics_cache: dict[str, tuple[float, RunStatistics]] = {}
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                created_at REAL NOT NULL, ended_at REAL, config_json TEXT NOT NULL,
                line_ids_json TEXT NOT NULL, dataset_fingerprint TEXT NOT NULL,
                skipped_count INTEGER NOT NULL DEFAULT 0, error TEXT,
                audio_bytes_saved INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL, line_id TEXT NOT NULL, speaker TEXT NOT NULL,
                text TEXT NOT NULL, reference_text_needs_review INTEGER NOT NULL,
                status TEXT NOT NULL, started_at REAL NOT NULL, finished_at REAL,
                latency_ms REAL, scheduling_delay_ms REAL NOT NULL, http_status INTEGER,
                service_request_id TEXT, finish_reason TEXT, error_type TEXT, error TEXT,
                audio_duration_seconds REAL, audio_bytes INTEGER NOT NULL, audio_path TEXT
            );
            CREATE INDEX IF NOT EXISTS request_run_sequence ON requests(run_id, sequence DESC);
            CREATE INDEX IF NOT EXISTS request_run_status_latency ON requests(run_id, status, latency_ms);
            CREATE INDEX IF NOT EXISTS request_run_time ON requests(run_id, started_at);
            CREATE TABLE IF NOT EXISTS events (
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                timestamp REAL NOT NULL, level TEXT NOT NULL, message TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS event_run_time ON events(run_id, timestamp DESC);
            CREATE TABLE IF NOT EXISTS gpu_samples (
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                timestamp REAL NOT NULL, uuid TEXT NOT NULL, "index" TEXT NOT NULL,
                name TEXT NOT NULL, memory_used_mib REAL, memory_total_mib REAL,
                utilization_percent REAL, temperature_celsius REAL, power_watts REAL
            );
            CREATE INDEX IF NOT EXISTS gpu_run_time ON gpu_samples(run_id, timestamp);
        """
        )

    def recover_interrupted(self) -> None:
        """Keep abandoned attempts visible after a process restart."""
        now = time.time()
        with self.lock, self.connection:
            self.connection.execute(
                "UPDATE requests SET status='interrupted', finished_at=?, error_type='process_restart', "
                "error='Workbench stopped before the result was persisted' WHERE status='inflight'",
                (now,),
            )
            self.connection.execute(
                "INSERT INTO events SELECT id, ?, 'warning', 'Task interrupted by process restart' "
                "FROM runs WHERE status IN ('running', 'stopping')",
                (now,),
            )
            self.connection.execute(
                "UPDATE runs SET status='interrupted', ended_at=? WHERE status IN ('running', 'stopping')",
                (now,),
            )

    def create_run(self, run: RunRecord) -> None:
        """Persist configuration before dispatching any network requests."""
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO runs(id,name,status,created_at,config_json,line_ids_json,dataset_fingerprint) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    run.id,
                    run.name,
                    run.status,
                    run.created_at,
                    run.config.model_dump_json(),
                    json.dumps(run.line_ids),
                    run.dataset_fingerprint,
                ),
            )
            self.event(run.id, "info", "Task started")

    def get_run(self, run_id: str) -> RunRecord:
        """Validate persisted JSON before returning the task API representation."""
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            else:
                values = dict(row)
                values["config"] = RunConfig.model_validate_json(
                    values.pop("config_json")
                )
                values["line_ids"] = json.loads(values.pop("line_ids_json"))
                return RunRecord.model_validate(values)

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        """Keep the history selector bounded without dropping stored history."""
        with self.lock:
            rows = self.connection.execute(
                "SELECT id FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            return [self.get_run(row["id"]) for row in rows.fetchall()]

    def update_status(
        self, run_id: str, status: RunStatus, error: str | None = None
    ) -> None:
        """Record task completion separately from stopping dispatch."""
        ended_at = None if status in ACTIVE_STATUSES else time.time()
        with self.lock, self.connection:
            self.connection.execute(
                "UPDATE runs SET status=?, ended_at=?, error=? WHERE id=?",
                (status, ended_at, error, run_id),
            )
            self.statistics_cache.pop(run_id, None)
            self.event(run_id, "error" if error else "info", error or f"Task {status}")

    def add_skipped(self, run_id: str, count: int) -> None:
        """Count missed launch slots without materializing a backlog."""
        with self.lock, self.connection:
            self.connection.execute(
                "UPDATE runs SET skipped_count=skipped_count+? WHERE id=?",
                (count, run_id),
            )

    def save_request(self, record: RequestRecord) -> None:
        """Insert or finish a known attempt using the validated record schema."""
        values = record.model_dump()
        columns = ",".join(values)
        placeholders = ",".join("?" for _ in values)
        assignments = ",".join(
            f"{column}=excluded.{column}" for column in values if column != "id"
        )
        with self.lock, self.connection:
            self.connection.execute(
                f"INSERT INTO requests({columns}) VALUES({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {assignments}",
                tuple(values.values()),
            )

    def add_audio_bytes(self, run_id: str, audio_bytes: int) -> None:
        """Persist sampled audio usage for historical display."""
        with self.lock, self.connection:
            self.connection.execute(
                "UPDATE runs SET audio_bytes_saved=audio_bytes_saved+? WHERE id=?",
                (audio_bytes, run_id),
            )

    def event(self, run_id: str, level: str, message: str) -> None:
        """Keep task events distinct from the read-only TTS log file."""
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO events VALUES(?,?,?,?)",
                (run_id, time.time(), level, message[:4000]),
            )

    def events(self, run_id: str) -> list[EventRecord]:
        """Return a bounded, chronological event tail."""
        with self.lock:
            rows = self.connection.execute(
                "SELECT timestamp,level,message FROM events WHERE run_id=? ORDER BY timestamp DESC LIMIT 100",
                (run_id,),
            ).fetchall()
            return [EventRecord.model_validate(dict(row)) for row in reversed(rows)]

    def request_page(
        self, run_id: str, offset: int, limit: int, failures_only: bool
    ) -> RequestPage:
        """Use SQL pagination so day-long tests do not fill application memory."""
        clause = "run_id=?" + (
            " AND status NOT IN ('success','inflight')" if failures_only else ""
        )
        with self.lock:
            total = self.connection.execute(
                f"SELECT COUNT(*) FROM requests WHERE {clause}", (run_id,)
            ).fetchone()[0]
            rows = self.connection.execute(
                f"SELECT * FROM requests WHERE {clause} ORDER BY sequence DESC LIMIT ? OFFSET ?",
                (run_id, limit, offset),
            ).fetchall()
            return RequestPage(
                total=total,
                records=[RequestRecord.model_validate(dict(row)) for row in rows],
            )

    def get_request(self, request_id: str) -> RequestRecord:
        """Resolve sampled audio through a recorded request, never a caller path."""
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM requests WHERE id=?", (request_id,)
            ).fetchone()
            if row is None:
                raise KeyError(request_id)
            else:
                return RequestRecord.model_validate(dict(row))

    def statistics(self, run_id: str) -> RunStatistics:
        """Exclude failures from latency, successful throughput, and audio totals."""
        now = time.time()
        with self.lock:
            cached = self.statistics_cache.get(run_id)
            if cached and now - cached[0] < 2:
                return cached[1]
            else:
                pass
            run = self.get_run(run_id)
            row = self.connection.execute(
                """
                SELECT COUNT(*) AS sent,
                  COALESCE(SUM(status='inflight'),0) AS inflight,
                  COALESCE(SUM(status='success'),0) AS success,
                  COALESCE(SUM(status NOT IN ('inflight','success')),0) AS failed,
                  COALESCE(SUM(status='timeout'),0) AS timeout,
                  COALESCE(SUM(status='truncated'),0) AS truncated,
                  COALESCE(SUM(status='interrupted'),0) AS interrupted,
                  COALESCE(SUM(status='success' AND (finish_reason IS NULL OR finish_reason='unknown')),0) AS unknown_finish,
                  COUNT(DISTINCT line_id) AS covered_lines,
                  COUNT(DISTINCT CASE WHEN status='success' THEN line_id END) AS successful_lines,
                  AVG(CASE WHEN status='success' THEN latency_ms END) AS average_latency_ms,
                  COALESCE(SUM(CASE WHEN status='success' THEN audio_duration_seconds END),0) AS successful_audio_seconds,
                  COUNT(CASE WHEN status='success' THEN audio_duration_seconds END) AS measured_audio_count
                FROM requests WHERE run_id=?
            """,
                (run_id,),
            ).fetchone()
            statistics = RunStatistics.model_validate(dict(row))
            percentiles: list[float | None] = []
            for quantile in (0.50, 0.95, 0.99):
                if statistics.success:
                    offset = max(0, math.ceil(statistics.success * quantile) - 1)
                    value = self.connection.execute(
                        "SELECT latency_ms FROM requests WHERE run_id=? AND status='success' "
                        "ORDER BY latency_ms LIMIT 1 OFFSET ?",
                        (run_id, offset),
                    ).fetchone()[0]
                    percentiles.append(value)
                else:
                    percentiles.append(None)
            (
                statistics.p50_latency_ms,
                statistics.p95_latency_ms,
                statistics.p99_latency_ms,
            ) = percentiles
            end = run.ended_at or now
            window_seconds = max(1, min(60, end - run.created_at))
            count = self.connection.execute(
                "SELECT COUNT(*) FROM requests WHERE run_id=? AND started_at>=?",
                (run_id, end - window_seconds),
            ).fetchone()[0]
            statistics.actual_requests_per_minute = count * 60 / window_seconds
            success_count = self.connection.execute(
                "SELECT COUNT(*) FROM requests WHERE run_id=? AND status='success' "
                "AND finished_at>=? AND finished_at<=?",
                (run_id, end - window_seconds, end),
            ).fetchone()[0]
            statistics.successful_requests_per_minute = (
                success_count * 60 / window_seconds
            )
            if len(self.statistics_cache) >= 64:
                self.statistics_cache.clear()
            else:
                pass
            self.statistics_cache[run_id] = (now, statistics)
            return statistics

    def timeline(self, run_id: str) -> list[TimelinePoint]:
        """Bound the number of chart points independently of task duration."""
        with self.lock:
            run = self.get_run(run_id)
            bucket_seconds = max(
                10, math.ceil(((run.ended_at or time.time()) - run.created_at) / 240)
            )
            rows = self.connection.execute(
                """
                SELECT CAST(started_at / ? AS INTEGER) * ? AS timestamp, COUNT(*) AS sent,
                  SUM(status='success') AS success, SUM(status NOT IN ('success','inflight')) AS failed,
                  AVG(CASE WHEN status='success' THEN latency_ms END) AS average_latency_ms
                FROM requests WHERE run_id=? GROUP BY timestamp ORDER BY timestamp
            """,
                (bucket_seconds, bucket_seconds, run_id),
            ).fetchall()
            return [TimelinePoint.model_validate(dict(row)) for row in rows]

    def save_gpu(self, run_id: str, samples: list[GpuSample]) -> None:
        """Store GPU samples only while a task is active."""
        with self.lock, self.connection:
            self.connection.executemany(
                "INSERT INTO gpu_samples VALUES(?,?,?,?,?,?,?,?,?,?)",
                [(run_id, *sample.model_dump().values()) for sample in samples],
            )

    def gpu_history(self, run_id: str) -> list[GpuSample]:
        """Downsample historical GPU observations to at most 241 points per card."""
        with self.lock:
            run = self.get_run(run_id)
            bucket_seconds = max(
                2, math.ceil(((run.ended_at or time.time()) - run.created_at) / 240)
            )
            rows = self.connection.execute(
                """
                SELECT CAST(timestamp / ? AS INTEGER) * ? AS timestamp, uuid,
                  MAX("index") AS "index", MAX(name) AS name,
                  AVG(memory_used_mib) AS memory_used_mib, MAX(memory_total_mib) AS memory_total_mib,
                  AVG(utilization_percent) AS utilization_percent,
                  AVG(temperature_celsius) AS temperature_celsius, AVG(power_watts) AS power_watts
                FROM gpu_samples WHERE run_id=? GROUP BY uuid, CAST(timestamp / ? AS INTEGER)
                ORDER BY timestamp
            """,
                (bucket_seconds, bucket_seconds, run_id, bucket_seconds),
            ).fetchall()
            return [GpuSample.model_validate(dict(row)) for row in rows]

    def completed_ids(self) -> list[str]:
        """Identify only terminal tasks eligible for bulk cleanup."""
        with self.lock:
            return [
                row[0]
                for row in self.connection.execute(
                    "SELECT id FROM runs WHERE status NOT IN ('running','stopping')"
                )
            ]

    def delete_run(self, run_id: str) -> None:
        """Cascade only a finished task's database rows."""
        with self.lock, self.connection:
            run = self.get_run(run_id)
            if run.status in ACTIVE_STATUSES:
                raise ValueError("Stop the task before clearing its results")
            else:
                self.connection.execute("DELETE FROM runs WHERE id=?", (run_id,))
                self.statistics_cache.pop(run_id, None)

    def close(self) -> None:
        """Release the database after background tasks have stopped."""
        with self.lock:
            self.connection.close()
