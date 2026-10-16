"""Retain exact HTTP inputs while sharing repeated reference-audio payloads."""

import hashlib
import json
import sqlite3
from threading import RLock
from typing import Literal

from pydantic import BaseModel, Field


class HttpRequestDetails(BaseModel):
    """Expose request inputs without copying authentication credentials."""

    method: Literal["POST"] = "POST"
    url: str
    headers: dict[str, str] = Field(
        default_factory=lambda: {"Content-Type": "application/json"}
    )
    body: dict[str, object]
    source: Literal["captured", "reconstructed"] = "captured"
    note: str = "请求发起前保存的参数快照；认证信息不展示。"


class HttpSnapshots:
    """Use the workbench transaction lock and cascade with its request records."""

    def __init__(self, connection: sqlite3.Connection, lock: RLock) -> None:
        """Add snapshot tables without changing existing task or request rows."""
        self.connection = connection
        self.lock = lock
        with self.lock, self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS http_payloads (
                    digest TEXT PRIMARY KEY, body_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS http_requests (
                    request_id TEXT PRIMARY KEY REFERENCES requests(id) ON DELETE CASCADE,
                    url TEXT NOT NULL,
                    payload_digest TEXT NOT NULL REFERENCES http_payloads(digest)
                );
                CREATE INDEX IF NOT EXISTS http_request_payload ON http_requests(payload_digest);
                """
            )

    def save(self, request_id: str, url: str, body: dict[str, object]) -> None:
        """Persist actual JSON values once per distinct body before sending HTTP."""
        body_json = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO http_payloads(digest,body_json) VALUES(?,?)",
                (digest, body_json),
            )
            self.connection.execute(
                "INSERT INTO http_requests(request_id,url,payload_digest) VALUES(?,?,?)",
                (request_id, url, digest),
            )

    def get(self, request_id: str) -> HttpRequestDetails | None:
        """Read a saved envelope; old requests deliberately have no invented snapshot."""
        with self.lock:
            row = self.connection.execute(
                "SELECT url,body_json FROM http_requests JOIN http_payloads "
                "ON payload_digest=digest WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row is None:
                return None
            else:
                return HttpRequestDetails(
                    url=row["url"], body=json.loads(row["body_json"])
                )

    def cleanup(self) -> None:
        """Remove payloads only after every request referring to them was cleared."""
        with self.lock, self.connection:
            self.connection.execute(
                "DELETE FROM http_payloads WHERE NOT EXISTS "
                "(SELECT 1 FROM http_requests WHERE payload_digest=digest)"
            )
