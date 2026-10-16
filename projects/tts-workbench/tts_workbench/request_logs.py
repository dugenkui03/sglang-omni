"""Read complete records for one service request from the configured log file."""

import hashlib
import os
import re
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from tts_workbench.monitor import MAX_LOG_BYTES, LogTail

LOG_HEADER = re.compile(
    r"^\[\d{2}:\d{2}:\d{2}\] (?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\b"
)
REQUEST_IDS = re.compile(r"^  request_ids?=(.+)$", re.MULTILINE)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
DOWNLOAD_CHUNK_BYTES = 64 * 1024


def log_records(stream: BinaryIO, end_offset: int) -> Iterator[tuple[int, str]]:
    """Yield whole formatted records, retaining an unfinished final record for polling."""
    lines: list[str] = []
    while stream.tell() < end_offset:
        offset = stream.tell()
        raw_line = stream.readline(end_offset - offset)
        if not raw_line:
            break
        line = ANSI_ESCAPE.sub("", raw_line.decode("utf-8", errors="replace"))
        if LOG_HEADER.match(line) and lines:
            yield offset, "".join(lines)
            lines = []
        lines.append(line)
    if lines and "".join(lines[-2:]).endswith("\n\n"):
        yield stream.tell(), "".join(lines)


def matches_request(record: str, request_id: str) -> bool:
    """Match exact single or batch identities without matching message substrings."""
    match = REQUEST_IDS.search(record)
    return bool(
        match
        and request_id
        in [value.strip() for value in match.group(1).strip("[]").split(",")]
    )


def tail_request_log(
    path: Path | None, cursor: str | None, request_id: str, limit: int
) -> LogTail:
    """Search the full file once, then follow complete records using a scoped cursor."""
    if path is None:
        return LogTail(source=None, error="No TTS log file was configured")
    try:
        with path.open("rb") as stream:
            metadata = os.fstat(stream.fileno())
            identity = f"{metadata.st_dev}:{metadata.st_ino}"
            filter_hash = hashlib.sha256(request_id.encode()).hexdigest()[:16]
            offset = 0
            reset = True
            if cursor:
                parts = cursor.split(":")
                if len(parts) != 5:
                    raise ValueError("Invalid request log cursor")
                previous_offset = int(parts[2])
                if (
                    ":".join(parts[:2]) == identity
                    and parts[4] == filter_hash
                    and 0 <= previous_offset <= metadata.st_size
                ):
                    stream.seek(max(0, previous_offset - 32))
                    fingerprint = hashlib.sha256(
                        stream.read(min(32, previous_offset))
                    ).hexdigest()[:16]
                    if fingerprint == parts[3]:
                        offset = previous_offset
                        reset = False
            stream.seek(offset)
            retained: deque[tuple[str, int, int]] = deque()
            line_count = byte_count = 0
            truncated = False
            for offset, record in log_records(stream, metadata.st_size):
                if matches_request(record, request_id):
                    record_lines = len(record.splitlines())
                    record_bytes = len(record.encode())
                    retained.append((record, record_lines, record_bytes))
                    line_count += record_lines
                    byte_count += record_bytes
                    while retained and (
                        line_count > limit or byte_count > MAX_LOG_BYTES
                    ):
                        _, removed_lines, removed_bytes = retained.popleft()
                        line_count -= removed_lines
                        byte_count -= removed_bytes
                        truncated = True
            stream.seek(max(0, offset - 32))
            fingerprint = hashlib.sha256(stream.read(min(32, offset))).hexdigest()[:16]
            return LogTail(
                source=path.name,
                cursor=f"{identity}:{offset}:{fingerprint}:{filter_hash}",
                text="".join(record for record, _, _ in retained),
                reset=reset or truncated,
                truncated=truncated,
            )
    except (OSError, ValueError) as exception:
        return LogTail(source=path.name, error=f"Log file unavailable: {exception}")


def download_request_log(path: Path, request_id: str) -> Iterator[bytes]:
    """Stream every matching record present at download start without a display limit."""
    with path.open("rb") as stream:
        end_offset = os.fstat(stream.fileno()).st_size
        output = bytearray()
        for _, record in log_records(stream, end_offset):
            if matches_request(record, request_id):
                output.extend(record.encode())
                if len(output) >= DOWNLOAD_CHUNK_BYTES:
                    yield bytes(output)
                    output.clear()
        if output:
            yield bytes(output)
