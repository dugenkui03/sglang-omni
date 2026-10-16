"""Archive complete service log records and index their request identities in SQLite."""

import asyncio
import hashlib
import logging
import os
import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import BinaryIO, Literal

from tts_workbench.monitor import MAX_LOG_BYTES, LogTail

logger = logging.getLogger(__name__)
LOG_HEADER = re.compile(
    rb"^(?:\x1b\[[0-9;]*m)*\[\d{2}:\d{2}:\d{2}\] (?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\b",
    re.MULTILINE,
)
REQUEST_IDS = re.compile(rb"^  request_ids?=(.+)$", re.MULTILINE)
BODY_REQUEST_IDS = re.compile(
    rb"""(?:\brequest_id\s*=\s*|["']request_id["']\s*:\s*["'])([A-Za-z0-9_-]+)"""
)
ANSI_ESCAPE = re.compile(rb"\x1b\[[0-9;]*m")
SCAN_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 64 * 1024
SYNC_INTERVAL_SECONDS = 2


def record_request_ids(record: bytes) -> set[str]:
    """Index formatter labels and structured coordinator payloads with their original IDs."""
    match = REQUEST_IDS.search(record)
    identities = set(match.group(1).strip(b"[]").split(b",")) if match else set()
    identities.update(BODY_REQUEST_IDS.findall(record))
    return {value.strip().decode("utf-8", errors="replace") for value in identities} - {
        "-",
        "",
    }


def log_records(stream: BinaryIO, end_offset: int) -> Iterator[tuple[int, bytes]]:
    """Keep complete multiline records across chunk boundaries and partial writes."""
    buffer = b""
    buffer_offset = stream.tell()
    remaining_bytes = end_offset - buffer_offset
    while remaining_bytes > 0:
        chunk = stream.read(min(SCAN_CHUNK_BYTES, remaining_bytes))
        if not chunk:
            break
        remaining_bytes -= len(chunk)
        buffer += chunk
        record_start = 0
        for header in LOG_HEADER.finditer(buffer):
            if header.start() > record_start:
                yield buffer_offset + header.start(), buffer[
                    record_start : header.start()
                ]
            record_start = header.start()
        buffer = buffer[record_start:]
        buffer_offset += record_start
    if ANSI_ESCAPE.sub(b"", buffer).endswith(b"\n\n") or (
        not LOG_HEADER.match(buffer) and buffer.endswith(b"\n")
    ):
        yield buffer_offset + len(buffer), buffer


class LogArchive:
    """Preserve log history separately from deletable test results and audio samples."""

    def __init__(self, root: Path, source: Path | None) -> None:
        """Create a WAL database with exact request-ID membership indexes."""
        root.mkdir(parents=True, exist_ok=True)
        self.database_path = root / "logs.sqlite3"
        self.source = source.resolve() if source is not None else None
        self.lock = threading.RLock()
        self.stop = asyncio.Event()
        self.error: str | None = None
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS log_sources (
                path TEXT PRIMARY KEY, identity TEXT NOT NULL,
                byte_offset INTEGER NOT NULL, fingerprint TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS log_records (
                id INTEGER PRIMARY KEY, source_path TEXT NOT NULL,
                text TEXT NOT NULL, line_count INTEGER NOT NULL, byte_count INTEGER NOT NULL,
                category TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS log_category_id ON log_records(category,id);
            CREATE TABLE IF NOT EXISTS log_totals (
                category TEXT PRIMARY KEY, line_count INTEGER NOT NULL, latest_id INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS log_requests (
                request_id TEXT NOT NULL, record_id INTEGER NOT NULL REFERENCES log_records(id),
                PRIMARY KEY(request_id, record_id)
            ) WITHOUT ROWID;
            """
        )
        if self.connection.execute("PRAGMA user_version").fetchone()[0] < 1:
            with self.connection:
                last_id = 0
                while True:
                    rows = self.connection.execute(
                        "SELECT id,text FROM log_records WHERE category='system' AND id>? ORDER BY id LIMIT 1000",
                        (last_id,),
                    ).fetchall()
                    if not rows:
                        break
                    last_id = rows[-1]["id"]
                    for row in rows:
                        identities = record_request_ids(row["text"].encode())
                        if identities:
                            self.connection.executemany(
                                "INSERT OR IGNORE INTO log_requests VALUES(?,?)",
                                [(identity, row["id"]) for identity in identities],
                            )
                            category = (
                                "inference"
                                if any(
                                    value.startswith("speech-") for value in identities
                                )
                                else "http"
                            )
                            self.connection.execute(
                                "UPDATE log_records SET category=? WHERE id=?",
                                (category, row["id"]),
                            )
                self.connection.execute("DELETE FROM log_totals")
                self.connection.execute(
                    "INSERT INTO log_totals SELECT category,SUM(line_count),MAX(id) FROM log_records GROUP BY category"
                )
                self.connection.execute("PRAGMA user_version=1")

    def close(self) -> None:
        """Release the archive only after background ingestion and readers finish."""
        with self.lock:
            self.connection.close()

    def synchronize(self) -> None:
        """Atomically save appended records and their byte checkpoint without reindexing history."""
        if self.source is None:
            self.error = "No TTS log file was configured"
            return
        try:
            with self.lock, self.connection, self.source.open("rb") as stream:
                metadata = os.fstat(stream.fileno())
                identity = f"{metadata.st_dev}:{metadata.st_ino}"
                source_path = str(self.source)
                previous = self.connection.execute(
                    "SELECT * FROM log_sources WHERE path=?", (source_path,)
                ).fetchone()
                offset = 0
                if (
                    previous is not None
                    and previous["identity"] == identity
                    and previous["byte_offset"] <= metadata.st_size
                ):
                    previous_offset = previous["byte_offset"]
                    stream.seek(max(0, previous_offset - 32))
                    fingerprint = hashlib.sha256(
                        stream.read(min(32, previous_offset))
                    ).hexdigest()
                    if fingerprint == previous["fingerprint"]:
                        offset = previous_offset
                stream.seek(offset)
                category_totals: dict[str, tuple[int, int]] = {}
                for offset, raw_record in log_records(stream, metadata.st_size):
                    raw_record = ANSI_ESCAPE.sub(b"", raw_record)
                    text = raw_record.decode("utf-8", errors="replace")
                    identities = record_request_ids(raw_record)
                    category = (
                        "inference"
                        if any(value.startswith("speech-") for value in identities)
                        else "http" if identities else "system"
                    )
                    line_count = len(text.splitlines())
                    inserted = self.connection.execute(
                        "INSERT INTO log_records(source_path,text,line_count,byte_count,category) VALUES(?,?,?,?,?)",
                        (source_path, text, line_count, len(raw_record), category),
                    )
                    record_id = inserted.lastrowid
                    assert record_id is not None
                    category_totals[category] = (
                        category_totals.get(category, (0, 0))[0] + line_count,
                        record_id,
                    )
                    if identities:
                        self.connection.executemany(
                            "INSERT INTO log_requests(request_id,record_id) VALUES(?,?)",
                            [(request_id, record_id) for request_id in identities],
                        )
                self.connection.executemany(
                    "INSERT INTO log_totals VALUES(?,?,?) ON CONFLICT(category) DO UPDATE SET "
                    "line_count=log_totals.line_count+excluded.line_count,latest_id=excluded.latest_id",
                    [
                        (category, count, latest)
                        for category, (count, latest) in category_totals.items()
                    ],
                )
                stream.seek(max(0, offset - 32))
                fingerprint = hashlib.sha256(stream.read(min(32, offset))).hexdigest()
                self.connection.execute(
                    "INSERT INTO log_sources VALUES(?,?,?,?) ON CONFLICT(path) DO UPDATE SET "
                    "identity=excluded.identity,byte_offset=excluded.byte_offset,fingerprint=excluded.fingerprint",
                    (source_path, identity, offset, fingerprint),
                )
                self.error = None
        except OSError as exception:
            message = f"Log ingestion unavailable: {exception}"
            if self.error != message:
                logger.warning(message)
            self.error = message

    async def run(self) -> None:
        """Ingest independently of browser tabs so history survives log rotation."""
        while not self.stop.is_set():
            try:
                await asyncio.to_thread(self.synchronize)
            except sqlite3.Error as exception:
                self.error = f"Log archive unavailable: {exception}"
                logger.error(self.error)
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=SYNC_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass

    def tail(
        self,
        cursor: str | None,
        request_id: str | None,
        limit: int,
        scope: Literal["all", "system", "inference"] = "all",
        keyword: str = "",
    ) -> LogTail:
        """Query all archived history; only the returned display window is bounded."""
        self.synchronize()
        filter_hash = hashlib.sha256(
            f"{scope}:{request_id or ''}:{keyword}".encode()
        ).hexdigest()[:16]
        offset = 0
        reset = True
        if cursor:
            parts = cursor.split(":")
            if len(parts) == 3 and parts[0] == "sqlite" and parts[2] == filter_hash:
                offset = int(parts[1])
                reset = False
        with self.lock:
            source = "FROM log_records AS records"
            conditions: list[str] = []
            parameters: list[str | int] = []
            if request_id:
                source += (
                    " JOIN log_requests AS requests ON records.id=requests.record_id"
                )
                conditions.append("requests.request_id=?")
                parameters.append(request_id)
            if scope != "all":
                conditions.append("category=?")
                parameters.append(scope)
            if keyword:
                conditions.append("instr(lower(records.text),lower(?))>0")
                parameters.append(keyword)
            predicate = " AND ".join(conditions) or "1=1"
            if request_id or keyword:
                totals = self.connection.execute(
                    f"SELECT COALESCE(MAX(records.id),0),COALESCE(SUM(records.line_count),0) {source} WHERE {predicate}",
                    parameters,
                ).fetchone()
            else:
                totals = self.connection.execute(
                    f"SELECT COALESCE(MAX(latest_id),0),COALESCE(SUM(line_count),0) FROM log_totals WHERE {predicate}",
                    parameters,
                ).fetchone()
            latest_id, matched_lines = totals
            if offset > latest_id:
                offset = 0
                reset = True
            rows = self.connection.execute(
                f"SELECT records.* {source} WHERE {predicate} AND records.id>? ORDER BY records.id DESC",
                [*parameters, offset],
            )
            retained: list[str] = []
            line_count = byte_count = 0
            truncated = False
            for row in rows:
                if (
                    line_count + row["line_count"] > limit
                    or byte_count + row["byte_count"] > MAX_LOG_BYTES
                ):
                    truncated = True
                    break
                retained.append(row["text"])
                line_count += row["line_count"]
                byte_count += row["byte_count"]
            return LogTail(
                source=f"{self.source.name if self.source else '服务日志'} · SQLite 历史",
                cursor=f"sqlite:{latest_id}:{filter_hash}",
                text="".join(reversed(retained)),
                reset=reset or truncated,
                truncated=truncated,
                matched_lines=matched_lines,
                archive_warning=self.error,
            )

    def download(self, request_id: str) -> Iterator[bytes]:
        """Stream the complete indexed history from a read snapshot without blocking ingestion."""
        self.synchronize()
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(
                "SELECT records.text FROM log_requests AS requests "
                "JOIN log_records AS records ON records.id=requests.record_id "
                "WHERE requests.request_id=? ORDER BY records.id",
                (request_id,),
            )
            output = bytearray()
            for row in rows:
                output.extend(row[0].encode())
                if len(output) >= DOWNLOAD_CHUNK_BYTES:
                    yield bytes(output)
                    output.clear()
            if output:
                yield bytes(output)
