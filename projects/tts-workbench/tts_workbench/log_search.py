"""对完整日志记录建立搜索索引，并限制历史日志保留行数。"""

import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

MAX_ARCHIVE_LINES = 20_000_000
SEARCH_PAGE_RECORDS = 100
SECONDS_PER_DAY = 86400
LOG_CLOCK_PATTERN = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\] ")
LOCATION_PATTERN = re.compile(r"^  ([^\s]+:\d+)\s*$", re.MULTILINE)
HEADER_PATTERN = re.compile(r"^\[([^\]]+)\] (\w+)")
METHOD_PATTERN = re.compile(r"^    method：(.*)$", re.MULTILINE)
MESSAGE_PATTERN = re.compile(r"^      msg：(.*)$", re.MULTILINE)


class LogRecordView(BaseModel):
    """提供代码位置、请求 ID 和可展开的完整日志内容。"""

    id: int
    time: str
    level: str
    location: str
    method: str
    summary: str
    request_ids: list[str]
    line_count: int
    text: str


class LogSearchPage(BaseModel):
    """使用最大记录 ID 固定搜索快照，避免新日志干扰翻页。"""

    records: list[LogRecordView]
    snapshot: int
    next_before: int | None
    page: int
    total_pages: int
    matched_records: int
    matched_lines: int
    retained_lines: int
    retention_limit: int = MAX_ARCHIVE_LINES
    warning: str | None = None


def log_time_seconds(text: str, previous_seconds: int) -> int:
    """按日志头的时分秒建立时间顺序，跨午夜延续日期，无时间的输出跟随前条。"""
    clock = LOG_CLOCK_PATTERN.match(text)
    if clock is None:
        return previous_seconds
    else:
        hour, minute, second = map(int, clock.groups())
        timestamp_seconds = (
            previous_seconds // SECONDS_PER_DAY * SECONDS_PER_DAY
            + hour * 3600
            + minute * 60
            + second
        )
        if timestamp_seconds < previous_seconds - SECONDS_PER_DAY // 2:
            timestamp_seconds += SECONDS_PER_DAY
        else:
            pass
        return timestamp_seconds


def prune_log_history(connection: sqlite3.Connection, limit: int) -> None:
    """按完整记录清理最旧日志及关联索引，释放的数据库页面供后续复用。"""
    total = connection.execute(
        "SELECT COALESCE(SUM(line_count),0) FROM log_totals"
    ).fetchone()[0]
    excess = total - limit
    if excess <= 0:
        return
    else:
        pass
    cutoff = 0
    for row in connection.execute("SELECT id,line_count FROM log_records ORDER BY id"):
        cutoff = row[0]
        excess -= row[1]
        if excess <= 0:
            break
        else:
            pass
    removed = connection.execute(
        "SELECT category,SUM(line_count) FROM log_records WHERE id<=? GROUP BY category",
        (cutoff,),
    ).fetchall()
    connection.execute("DELETE FROM log_requests WHERE record_id<=?", (cutoff,))
    connection.execute("DELETE FROM log_records WHERE id<=?", (cutoff,))
    connection.executemany(
        "UPDATE log_totals SET line_count=line_count-? WHERE category=?",
        [(row[1], row[0]) for row in removed],
    )
    connection.execute("DELETE FROM log_totals WHERE line_count<=0")


def initialize_log_search(connection: sqlite3.Connection, retention_limit: int) -> None:
    """为新日志库建立索引，重启时复用已有索引。"""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(log_records)")}
    if "sort_seconds" not in columns:
        with connection:
            connection.execute("BEGIN")
            connection.execute(
                "ALTER TABLE log_records ADD COLUMN sort_seconds INTEGER NOT NULL DEFAULT 0"
            )
            previous_seconds = 0
            for row in connection.execute(
                "SELECT id,text FROM log_records ORDER BY id"
            ):
                timestamp_seconds = log_time_seconds(row[1], previous_seconds)
                connection.execute(
                    "UPDATE log_records SET sort_seconds=? WHERE id=?",
                    (timestamp_seconds, row[0]),
                )
                previous_seconds = max(previous_seconds, timestamp_seconds)
    else:
        pass
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS log_request_record ON log_requests(record_id);
        CREATE INDEX IF NOT EXISTS log_location_id ON log_records(location,id);
        CREATE INDEX IF NOT EXISTS log_time_id ON log_records(sort_seconds,id);
        CREATE VIRTUAL TABLE IF NOT EXISTS log_text_search USING fts5(
            text,content='log_records',content_rowid='id',tokenize='trigram',detail='none'
        );
        CREATE TRIGGER IF NOT EXISTS log_search_insert AFTER INSERT ON log_records BEGIN
            INSERT INTO log_text_search(rowid,text) VALUES(new.id,new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS log_search_delete AFTER DELETE ON log_records BEGIN
            INSERT INTO log_text_search(log_text_search,rowid,text) VALUES('delete',old.id,old.text);
        END;
        """
    )
    with connection:
        prune_log_history(connection, retention_limit)


def search_log_history(
    database_path: Path,
    request_id: str = "",
    keyword: str = "",
    location: str = "",
    scope: Literal["all", "system", "inference"] = "all",
    limit: int = 2000,
    snapshot: int | None = None,
    before: int | None = None,
    page: int = 1,
    order: Literal["asc", "desc"] = "desc",
) -> LogSearchPage:
    """在固定快照内按页码搜索，行数预算和完整记录共同决定页边界。"""
    with closing(sqlite3.connect(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("BEGIN")
        latest_id = connection.execute(
            "SELECT COALESCE(MAX(id),0) FROM log_records"
        ).fetchone()[0]
        snapshot = min(snapshot, latest_id) if snapshot is not None else latest_id
        predicates = ["records.id<=?"]
        parameters: list[str | int] = [snapshot]
        source = "FROM log_records AS records"
        if request_id:
            source += " JOIN log_requests AS requests ON records.id=requests.record_id"
            predicates.append("requests.request_id=?")
            parameters.append(request_id)
        else:
            pass
        if keyword:
            if len(keyword) >= 3:
                source += " JOIN log_text_search ON log_text_search.rowid=records.id"
                predicates.append("log_text_search.text LIKE ?")
                parameters.append(f"%{keyword}%")
            else:
                pass
            predicates.append("instr(lower(records.text),lower(?))>0")
            parameters.append(keyword)
        else:
            pass
        if location:
            if re.search(r":\d+$", location):
                predicates.append("records.location=?")
                parameters.append(location)
            else:
                predicates.append("records.location LIKE ?")
                parameters.append(f"{location}%")
        else:
            pass
        if scope != "all":
            predicates.append("records.category=?")
            parameters.append(scope)
        else:
            pass
        where = " AND ".join(predicates)
        direction = "ASC" if order == "asc" else "DESC"
        order_by = f"records.sort_seconds {direction},records.id {direction}"
        boundaries = connection.execute(
            f"SELECT records.id,records.line_count,records.sort_seconds {source} WHERE {where} ORDER BY {order_by}",
            parameters,
        )
        matched_records = matched_lines = total_pages = 0
        page_records = page_lines = 0
        selected_start = None
        last_start = (0, 0)
        cursor_page = None
        # 仅扫描时间、记录 ID 和行数，避免为跳页加载前面所有日志正文。
        for boundary in boundaries:
            if not page_records or (
                page_records >= SEARCH_PAGE_RECORDS
                or page_lines + boundary["line_count"] > limit
            ):
                total_pages += 1
                page_records = page_lines = 0
                last_start = (boundary["sort_seconds"], boundary["id"])
                if total_pages == page:
                    selected_start = last_start
                else:
                    pass
            else:
                pass
            if before is not None and cursor_page is None and boundary["id"] < before:
                cursor_page = total_pages
            else:
                pass
            page_records += 1
            page_lines += boundary["line_count"]
            matched_records += 1
            matched_lines += boundary["line_count"]
        if before is not None:
            where += " AND records.id<?"
            parameters.append(before)
            page = cursor_page or max(1, total_pages)
        else:
            page = max(1, min(page, total_pages))
            comparison = ">=" if order == "asc" else "<="
            where += f" AND (records.sort_seconds,records.id){comparison}(?,?)"
            parameters.extend(
                selected_start if selected_start is not None else last_start
            )
        rows = connection.execute(
            f"SELECT records.* {source} WHERE {where} ORDER BY {order_by} LIMIT ?",
            [*parameters, SEARCH_PAGE_RECORDS + 1],
        )
        records: list[LogRecordView] = []
        displayed_lines = 0
        next_before = None
        for row in rows:
            if records and (
                displayed_lines + row["line_count"] > limit
                or len(records) >= SEARCH_PAGE_RECORDS
            ):
                next_before = records[-1].id
                break
            else:
                pass
            text = row["text"]
            header = HEADER_PATTERN.search(text)
            method = METHOD_PATTERN.search(text)
            message = MESSAGE_PATTERN.search(text)
            identities = connection.execute(
                "SELECT request_id FROM log_requests WHERE record_id=? ORDER BY request_id",
                (row["id"],),
            ).fetchall()
            records.append(
                LogRecordView(
                    id=row["id"],
                    time=header.group(1) if header else "",
                    level=header.group(2) if header else "SYSTEM",
                    location=row["location"],
                    method=method.group(1) if method else "",
                    summary=(message.group(1) if message else text.splitlines()[0])[
                        :240
                    ],
                    request_ids=[identity[0] for identity in identities],
                    line_count=row["line_count"],
                    text=text,
                )
            )
            displayed_lines += row["line_count"]
        retained_lines = connection.execute(
            "SELECT COALESCE(SUM(line_count),0) FROM log_totals"
        ).fetchone()[0]
        return LogSearchPage(
            records=records,
            snapshot=snapshot,
            next_before=next_before,
            page=page,
            total_pages=total_pages,
            matched_records=matched_records,
            matched_lines=matched_lines,
            retained_lines=retained_lines,
        )
