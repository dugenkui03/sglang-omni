"""Verify authentication, data boundaries, and log cursor behavior."""

import asyncio
import base64
import time
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tts_workbench.app import create_app
from tts_workbench.config import RunConfig, Settings
from tts_workbench.log_search import search_log_history
from tts_workbench.monitor import (
    DEFAULT_LOG_LINES,
    MAX_LOG_BYTES,
    MAX_LOG_LINES,
    optional_number,
    tail_log,
)
from tts_workbench.records import GpuSample, RequestRecord, RunRecord
from tts_workbench.request_logs import LogArchive
from tts_workbench.storage import Store


def test_authenticated_api_and_static_files(settings: Settings) -> None:
    """Protect static assets and reject cross-origin or untagged mutations."""
    app = create_app(replace(settings, password="synthetic-test-password"))
    with TestClient(app) as client:
        assert client.get("/").status_code == 401
        assert client.get("/static/app.js").status_code == 401
        client.headers["Authorization"] = (
            "Basic " + base64.b64encode(b"workbench:synthetic-test-password").decode()
        )
        assert client.get("/").status_code == 200
        assert client.get("/api/workbench/dataset").json()["line_count"] == 2
        assert client.get("/api/workbench/task-name").json() == {
            "name": "qwen3-tts-1.7b-base-001"
        }
        assert (
            client.get("/api/workbench/references/unit_0").headers["content-type"]
            == "audio/wav"
        )
        assert client.delete("/api/workbench/runs/completed").status_code == 403
        client.headers["X-Workbench-Client"] = "1"
        assert (
            client.delete(
                "/api/workbench/runs/completed",
                headers={"Origin": "https://unrelated.test"},
            ).status_code
            == 403
        )
        assert client.delete("/api/workbench/runs/completed").json() == {"cleared": 0}
        assert client.get("/api/workbench/runs/unknown").status_code == 404
        assert client.get("/api/workbench/audio/unknown").status_code == 404
        assert (
            client.get("/api/workbench/logs").json()["text"] == "original service log\n"
        )


def test_single_owner_lock(settings: Settings) -> None:
    """Prevent a second process from marking a live test interrupted."""
    with TestClient(create_app(settings)):
        with pytest.raises(RuntimeError, match="Another workbench"):
            with TestClient(create_app(settings)):
                pytest.fail("A second owner must not start")


def test_invalid_selection_is_rejected_without_task(settings: Settings) -> None:
    """Reject mismatched single-line selections before persisting a workload."""
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/workbench/runs",
            json={"model": "test", "mode": "single"},
            headers={"X-Workbench-Client": "1"},
        )
        assert response.status_code == 400
        assert client.get("/api/workbench/runs").json()["runs"] == []


def test_log_append_rotation_truncation_and_missing(tmp_path: Path) -> None:
    """Never lose an appended line or confuse a replacement log with its predecessor."""
    path = tmp_path / "server.log"
    assert tail_log(path, None).error
    path.write_text("开始\n", encoding="utf-8")
    first = tail_log(path, None)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("新的台词\n")
    appended = tail_log(path, first.cursor)
    assert appended.text == "新的台词\n" and not appended.reset
    path.write_text("截断后重新增长的日志" * 10, encoding="utf-8")
    truncated = tail_log(path, appended.cursor)
    assert truncated.reset and truncated.text.startswith("截断")
    path.rename(tmp_path / "server.log.1")
    path.write_text("新文件\n", encoding="utf-8")
    rotated = tail_log(path, truncated.cursor)
    assert rotated.reset and rotated.text == "新文件\n"
    assert not tail_log(path, rotated.cursor).text


def test_log_multibyte_read_boundary(tmp_path: Path) -> None:
    """Keep an incomplete UTF-8 character for the next incremental read."""
    path = tmp_path / "unicode.log"
    path.write_bytes(b"")
    empty = tail_log(path, None)
    source = "a" * (MAX_LOG_BYTES - 1) + "中文"
    path.write_text(source, encoding="utf-8")
    first = tail_log(path, empty.cursor)
    second = tail_log(path, first.cursor)
    assert first.text + second.text == source


def test_log_line_limit_and_catching_up(tmp_path: Path) -> None:
    """Respect default and maximum windows while discarding stale byte backlogs."""
    path = tmp_path / "large.log"
    path.write_text("".join(f"line {index}\n" for index in range(35000)))
    first = tail_log(path, None)
    assert len(first.text.splitlines()) == DEFAULT_LOG_LINES
    assert first.text.startswith("line 33000\n")
    maximum = tail_log(path, None, MAX_LOG_LINES)
    assert len(maximum.text.splitlines()) == MAX_LOG_LINES
    assert maximum.text.startswith("line 5000\n")
    smaller = tail_log(path, None, 100)
    assert len(smaller.text.splitlines()) == 100
    assert smaller.text.startswith("line 34900\n")
    with path.open("a") as stream:
        stream.write("new\n" * (MAX_LOG_BYTES // 2 + 2))
        stream.write("latest\n")
    caught_up = tail_log(path, first.cursor, 50)
    assert caught_up.reset
    assert len(caught_up.text.splitlines()) == 50
    assert caught_up.text.endswith("latest\n")


def request_record(identity: str, message: str) -> str:
    """Build the production multiline shape for request-log boundary cases."""
    return (
        f"[02:00:47] INFO\n  {identity}\n  sglang_model.py:315\n"
        f"    method：Qwen3TTSTalkerDecoderLayer.forward()\n      msg：{message}\n\n"
    )


def test_request_logs_keep_records_and_follow_partial_append(tmp_path: Path) -> None:
    """Match exact batch members and preserve counts independently of the display cap."""
    path = tmp_path / "server.log"
    first = request_record("request_id=speech-a", "开始注意力计算，layer_id=12")
    other = request_record("request_id=speech-a-extra", "mentions speech-a")
    batch = request_record("request_ids=[speech-b, speech-a]", "合批计算")
    path.write_text(first + other + batch[:-4])
    with closing(LogArchive(tmp_path, path)) as archive:
        initial = archive.tail(None, "speech-a", 20000)
        assert initial.text == first
        assert initial.matched_lines == len(first.splitlines())
        with path.open("a") as stream:
            stream.write(batch[-4:])
        appended = archive.tail(initial.cursor, "speech-a", 20000)
        assert appended.text == batch and not appended.reset
        assert appended.matched_lines == len((first + batch).splitlines())
        assert archive.tail(appended.cursor, "speech-a", 20000).text == ""
        smaller = archive.tail(None, "speech-a", 7)
        assert smaller.text == batch and smaller.truncated
        assert smaller.matched_lines == appended.matched_lines
        assert b"".join(archive.download("speech-a")).decode() == first + batch
        changed = archive.tail(appended.cursor, "speech-a-extra", 20000)
        assert changed.reset and changed.text == other


def test_request_search_reads_history_outside_latest_byte_window(
    tmp_path: Path,
) -> None:
    """Find old requests outside both the latest byte window and the 20k-line display."""
    path = tmp_path / "large.log"
    old = request_record("request_id=speech-old", "historical request")
    recent = request_record("request_id=http-health", "health check")
    path.write_text(old + recent * (MAX_LOG_BYTES // len(recent.encode()) + 1))
    assert "speech-old" not in tail_log(path, None).text
    with closing(LogArchive(tmp_path, path)) as archive:
        assert archive.tail(None, "speech-old", 20000).text == old
        assert b"".join(archive.download("speech-old")).decode() == old


@pytest.mark.parametrize("chunk_bytes", [1, 17, 64])
def test_archive_chunk_boundaries_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunk_bytes: int
) -> None:
    """Preserve ANSI headers, UTF-8, partial writes, and persisted checkpoints."""
    monkeypatch.setattr("tts_workbench.request_logs.SCAN_CHUNK_BYTES", chunk_bytes)
    path = tmp_path / "server.log"
    first = request_record("request_id=speech-a", "中文边界")
    colored = "\x1b[36m" + first.replace("\n", "\x1b[0m\n")
    path.write_text(colored)
    with closing(LogArchive(tmp_path, path)) as archive:
        assert archive.tail(None, "speech-a", 20000).text == first
    with closing(LogArchive(tmp_path, path)) as archive:
        assert b"".join(archive.download("speech-a")).decode() == first
        path.rename(tmp_path / "server.old.log")
        path.write_text(first.replace("中文边界", "重启后的请求"))
        assert archive.tail(None, "speech-a", 20000).matched_lines == 12
        path.unlink()
        historical = archive.tail(None, "speech-a", 20000)
        assert "中文边界" in historical.text and "重启后的请求" in historical.text
        assert historical.archive_warning


def test_archive_system_inference_and_truncate(tmp_path: Path) -> None:
    """Keep startup logs selectable and preserve history when the source is truncated."""
    path = tmp_path / "server.log"
    startup = request_record("request_id=-", "Application startup complete")
    speech = request_record("request_id=speech-a", "模型推理")
    health = request_record("request_id=http-a", "GET /health")
    path.write_text(startup + speech + health)
    with closing(LogArchive(tmp_path, path)) as archive:
        assert archive.tail(None, None, 20000, "system").text == startup
        assert archive.tail(None, None, 20000, "inference").text == speech
        assert archive.tail(None, None, 20000).text == startup + speech + health
        path.write_text(request_record("request_id=-", "Restarted"))
        assert (
            "Application startup complete"
            in archive.tail(None, None, 20000, "system").text
        )
        assert "Restarted" in archive.tail(None, None, 20000, "system").text


def test_request_log_api_and_unlimited_download(settings: Settings) -> None:
    """Expose complete filtered records and export beyond the chosen display cap."""
    first = request_record("request_id=speech-a", "first")
    second = request_record("request_id=speech-a", "second")
    assert settings.log_file is not None
    settings.log_file.write_text(first + second)
    with TestClient(create_app(settings)) as client:
        response = client.get(
            "/api/workbench/logs", params={"request_id": "speech-a", "limit": 7}
        )
        assert response.status_code == 200
        assert response.json()["text"] == second
        assert response.json()["truncated"]
        assert response.json()["matched_lines"] == 12
        download = client.get(
            "/api/workbench/logs/download", params={"request_id": "speech-a"}
        )
        assert download.status_code == 200
        assert download.text == first + second
        assert "attachment" in download.headers["content-disposition"]
        assert 'filename="speech-a.log"' in download.headers["content-disposition"]
        client.delete(
            "/api/workbench/runs/completed", headers={"X-Workbench-Client": "1"}
        )
        assert (
            client.get(
                "/api/workbench/logs/download", params={"request_id": "speech-a"}
            ).text
            == first + second
        )


def test_log_api_default_maximum_and_complete_download(settings: Settings) -> None:
    """Bound rendered windows while exporting a request larger than the maximum."""
    record = request_record("request_id=speech-a", "attention")
    history = record * 5001
    assert settings.log_file is not None
    settings.log_file.write_text(history)
    with TestClient(create_app(settings)) as client:
        default = client.get("/api/workbench/logs").json()
        assert 1900 < len(default["text"].splitlines()) <= DEFAULT_LOG_LINES
        assert default["matched_lines"] == 30006
        maximum = client.get("/api/workbench/logs", params={"limit": 30000})
        assert maximum.status_code == 200
        assert len(maximum.json()["text"].splitlines()) == MAX_LOG_LINES
        assert (
            client.get("/api/workbench/logs", params={"limit": 30001}).status_code
            == 422
        )
        download = client.get(
            "/api/workbench/logs/download", params={"request_id": "speech-a"}
        )
        assert download.text == history


def test_request_search_matches_task_names_with_pagination(settings: Settings) -> None:
    """Search across tasks, preserve literal wildcard characters and combine failure filtering."""
    with TestClient(create_app(settings)) as client:
        store = client.app.state.runtime.store
        for index, name in enumerate(
            ("负载_ALPHA_100%", "负载 beta", "other", "负载_ALPHA_100x")
        ):
            store.create_run(
                RunRecord(
                    id=f"run-{index}",
                    name=name,
                    status="completed",
                    created_at=index,
                    config=RunConfig(),
                    line_ids=["unit_0"],
                    dataset_fingerprint="test",
                )
            )
            for sequence, status in enumerate(("success", "failed")):
                store.save_request(
                    RequestRecord(
                        id=f"request-{index}-{sequence}",
                        run_id=f"run-{index}",
                        sequence=sequence,
                        line_id="unit_0",
                        speaker="test",
                        text="test",
                        reference_text_needs_review=False,
                        started_at=(4 - index) * 2 + sequence,
                        status=status,
                    )
                )
        first = client.get(
            "/api/workbench/requests", params={"task_name": "负载", "limit": 2}
        ).json()
        second = client.get(
            "/api/workbench/requests",
            params={"task_name": "负载", "limit": 2, "offset": 2},
        ).json()
        assert first["total"] == second["total"] == 6
        assert [row["run_id"] for row in first["records"]] == ["run-3", "run-3"]
        assert {row["id"] for row in first["records"]}.isdisjoint(
            row["id"] for row in second["records"]
        )
        assert all(
            "负载" in row["run_name"] for row in first["records"] + second["records"]
        )
        failures = client.get(
            "/api/workbench/requests",
            params={"task_name": "负载", "failures_only": True},
        ).json()
        assert failures["total"] == 3
        assert all(row["status"] == "failed" for row in failures["records"])
        literal = client.get(
            "/api/workbench/requests", params={"task_name": "_100%"}
        ).json()
        assert literal["total"] == 2
        assert all(row["run_name"] == "负载_ALPHA_100%" for row in literal["records"])


async def test_gpu_na_is_not_zero(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parse genuine CLI field shapes while preserving unsupported values."""

    class Process:
        """Provide process output only inside this isolated parser test."""

        returncode: int = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            """Return a CSV row matching the NVIDIA query contract."""
            return b"0, GPU-test, Test GPU, 1024, 8192, 42, 60, [N/A]\n", b""

    async def create_process(*command: str, stdout: int, stderr: int) -> Process:
        """Replace only subprocess creation during the device-parser assertion."""
        assert (
            command[0] == "nvidia-smi" and stdout == stderr == asyncio.subprocess.PIPE
        )
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    with TestClient(create_app(settings)) as client:
        runtime = client.app.state.runtime
        sample = await runtime.gpu.sample()
        assert sample.devices[0].utilization_percent == 42
        assert sample.devices[0].power_watts is None
        assert optional_number("[Not Supported]") is None


def test_recent_gpu_window_survives_restart_without_tasks(tmp_path: Path) -> None:
    """Keep device-wide telemetry for 30 minutes, including idle time and null metrics."""
    now = time.time()
    sample = GpuSample(
        timestamp=now,
        uuid="gpu-1",
        index="0",
        name="Test GPU",
        memory_used_mib=1024,
        memory_total_mib=4096,
        utilization_percent=0,
        temperature_celsius=42,
        power_watts=None,
    )
    with closing(Store(tmp_path)) as store:
        store.save_gpu(
            None, [sample.model_copy(update={"timestamp": now - 1801}), sample]
        )
    with closing(Store(tmp_path)) as store:
        history = store.recent_gpu_history()
        assert len(history) == 1
        assert history[0] == sample


def test_archive_indexes_payload_ids_and_searches_historical_keywords(
    tmp_path: Path,
) -> None:
    """Include coordinator payload IDs and startup records beyond the display window."""
    path = tmp_path / "server.log"
    startup = request_record("request_id=-", "Application startup complete")
    payload = request_record(
        "request_id=-", '{"request_id": "speech-a", "stage": "preprocess"}'
    )
    path.write_text(
        startup + payload + request_record("request_id=speech-other", "other") * 100
    )
    with closing(LogArchive(tmp_path, path)) as archive:
        assert archive.tail(None, "speech-a", 100).text == payload
        assert archive.tail(None, None, 100, "system", "STARTUP").text == startup
        assert archive.tail(None, None, 100, "inference", "preprocess").text == payload
        assert archive.tail(None, "speech-a-extra", 100).matched_lines == 0


def test_indexed_log_search_combines_filters_and_freezes_pages(tmp_path: Path) -> None:
    """验证全历史搜索、字面关键字匹配和快照分页不受新日志干扰。"""
    path = tmp_path / "server.log"
    old = request_record(
        "request_ids=[speech-a, speech-b]", '开始注意力计算 {"value":"100%_ok"}'
    )
    recent = request_record("request_id=speech-a-extra", "irrelevant")
    path.write_text(old * 3 + recent * 120)
    with closing(LogArchive(tmp_path, path)) as archive:
        archive.synchronize()
        page = search_log_history(
            archive.database_path, "speech-a", "注意力", "sglang_model.py:315", limit=6
        )
        assert page.matched_records == 3
        assert page.records[0].position == 1
        assert page.matched_lines == 18
        assert page.records[0].request_ids == ["speech-a", "speech-b"]
        assert page.records[0].text == old
        assert page.records[0].location == "sglang_model.py:315"
        assert page.next_before == page.records[-1].id
        with path.open("a") as stream:
            stream.write(old)
        archive.synchronize()
        second = search_log_history(
            archive.database_path,
            "speech-a",
            "注意力",
            "sglang_model.py:315",
            limit=6,
            snapshot=page.snapshot,
            before=page.next_before,
        )
        assert second.matched_records == 3
        assert second.records[0].position == 2
        assert second.records[0].id < page.records[0].id
        assert (
            search_log_history(archive.database_path, keyword="100%_ok").matched_records
            == 4
        )
        assert (
            search_log_history(
                archive.database_path, location="sglang_model.py:31"
            ).matched_records
            == 0
        )
        assert (
            search_log_history(
                archive.database_path, location="sglang_model.py"
            ).matched_records
            == 124
        )
        assert (
            search_log_history(archive.database_path, keyword="注意").matched_records
            == 4
        )


def test_log_retention_removes_whole_records_without_reimport(tmp_path: Path) -> None:
    """验证行数上限、索引清理及重启后不重复导入已清理记录。"""
    path = tmp_path / "server.log"
    records = [
        request_record(f"request_id=speech-{index}", f"message-{index}")
        for index in range(6)
    ]
    path.write_text("".join(records[:5]))
    with closing(LogArchive(tmp_path, path, retention_limit=14)) as archive:
        archive.synchronize()
        page = search_log_history(archive.database_path)
        assert page.retained_lines == 12
        assert page.matched_records == 2
        assert not list(archive.download("speech-0"))
        assert (
            search_log_history(
                archive.database_path, keyword="message-0"
            ).matched_records
            == 0
        )
        assert b"".join(archive.download("speech-4")).decode() == records[4]
    with closing(LogArchive(tmp_path, path, retention_limit=14)) as archive:
        archive.synchronize()
        assert search_log_history(archive.database_path).retained_lines == 12
        with path.open("a") as stream:
            stream.write(records[5])
        archive.synchronize()
        page = search_log_history(archive.database_path)
        assert [row.request_ids for row in page.records] == [["speech-5"], ["speech-4"]]
        assert (
            archive.connection.execute("SELECT COUNT(*) FROM log_requests").fetchone()[
                0
            ]
            == 2
        )
        assert (
            archive.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        )


def test_log_search_api_returns_complete_expandable_records(settings: Settings) -> None:
    """验证接口返回代码位置、请求 ID 及未被行数预算截断的完整记录。"""
    assert settings.log_file is not None
    record = request_record("request_id=speech-search", "search content")
    settings.log_file.write_text(record)
    with TestClient(create_app(settings)) as client:
        client.get("/api/workbench/logs")
        response = client.get(
            "/api/workbench/logs/search",
            params={"request_id": "speech-search", "keyword": "CONTENT", "limit": 1},
        )
        assert response.status_code == 200
        page = response.json()
        assert page["records"][0]["text"] == record
        assert page["records"][0]["request_ids"] == ["speech-search"]
        assert page["records"][0]["method"] == "Qwen3TTSTalkerDecoderLayer.forward()"
        assert page["retention_limit"] == 20_000_000


def test_log_search_jumps_to_numbered_pages_in_a_frozen_result(
    settings: Settings,
) -> None:
    """直接跳到第 35 页和末页，新日志不会导致重复、遗漏或虚增页数。"""
    assert settings.log_file is not None
    record = request_record("request_id=speech-pages", "分页关键字")
    settings.log_file.write_text(record * 3508)
    with TestClient(create_app(settings)) as client:
        client.get("/api/workbench/logs")
        query = {
            "request_id": "speech-pages",
            "keyword": "分页关键字",
            "location": "sglang_model.py",
        }
        first = client.get("/api/workbench/logs/search", params=query).json()
        assert first["total_pages"] == 36
        assert first["page"] == 1
        with settings.log_file.open("a") as stream:
            stream.write(record)
        client.get("/api/workbench/logs")
        query["snapshot"] = first["snapshot"]
        middle = client.get(
            "/api/workbench/logs/search", params={**query, "page": 35}
        ).json()
        assert middle["page"] == 35
        assert middle["matched_records"] == 3508
        assert [row["position"] for row in middle["records"]] == list(range(3401, 3501))
        assert [row["id"] for row in middle["records"]] == list(range(108, 8, -1))
        last = client.get(
            "/api/workbench/logs/search", params={**query, "page": 44}
        ).json()
        assert last["page"] == 36
        assert [row["position"] for row in last["records"]] == list(range(3501, 3509))
        assert [row["id"] for row in last["records"]] == list(range(8, 0, -1))
        assert last["next_before"] is None
        assert (
            client.get("/api/workbench/logs/search", params={"page": 0}).status_code
            == 422
        )


def test_log_page_boundaries_keep_oversized_records_whole(tmp_path: Path) -> None:
    """行数预算影响总页数，超长单条独占一页，直接跳页仍覆盖全部记录。"""
    path = tmp_path / "server.log"
    short = request_record("request_id=speech-budget", "short")
    long = request_record("request_id=speech-budget", "long\n" + "  extra\n" * 50)
    path.write_text(short * 2 + long + short * 2)
    with closing(LogArchive(tmp_path, path)) as archive:
        archive.synchronize()
        pages = [
            search_log_history(archive.database_path, limit=12, page=number)
            for number in (1, 2, 3)
        ]
        assert [page.total_pages for page in pages] == [3, 3, 3]
        assert [[record.position for record in page.records] for page in pages] == [
            [1, 2],
            [3],
            [4, 5],
        ]
        assert [[record.id for record in page.records] for page in pages] == [
            [5, 4],
            [3],
            [2, 1],
        ]
        assert pages[1].records[0].text == long
        empty = search_log_history(archive.database_path, request_id="missing", page=9)
        assert empty.total_pages == 0
        assert not empty.records


def test_log_time_sort_applies_before_pagination_and_survives_upgrade(
    tmp_path: Path,
) -> None:
    """验证乱序日志全量时间排序、同秒稳定顺序及旧库补建索引。"""
    path = tmp_path / "server.log"
    records = [
        request_record("request_id=speech-sort", f"message-{second}").replace(
            "[02:00:47]", f"[12:{second // 60:02}:{second % 60:02}]"
        )
        for second in reversed(range(205))
    ]
    path.write_text("".join(records))
    with closing(LogArchive(tmp_path, path)) as archive:
        archive.synchronize()
        with archive.connection:
            archive.connection.execute("DROP INDEX log_time_id")
            archive.connection.execute(
                "ALTER TABLE log_records DROP COLUMN sort_seconds"
            )
    with closing(LogArchive(tmp_path, path)) as archive:
        first = search_log_history(
            archive.database_path, request_id="speech-sort", order="asc"
        )
        last = search_log_history(
            archive.database_path, request_id="speech-sort", order="asc", page=3
        )
        newest = search_log_history(
            archive.database_path, request_id="speech-sort", order="desc"
        )
        assert first.total_pages == 3
        assert first.records[0].time == "12:00:00"
        assert first.records[-1].time == "12:01:39"
        assert last.records[0].time == "12:03:20"
        assert last.records[-1].time == "12:03:24"
        assert [record.position for record in last.records] == list(range(201, 206))
        assert newest.records[0].position == 1
        assert newest.records[0].id == last.records[-1].id
        with path.open("a") as stream:
            stream.write(
                request_record("request_id=speech-sort", "午夜之前").replace(
                    "[02:00:47]", "[23:59:59]"
                )
                + request_record("request_id=speech-sort", "午夜之后").replace(
                    "[02:00:47]", "[00:00:01]"
                )
                * 2
            )
        archive.synchronize()
        midnight = search_log_history(archive.database_path, order="desc")
        assert [record.time for record in midnight.records[:3]] == [
            "00:00:01",
            "00:00:01",
            "23:59:59",
        ]
        assert midnight.records[0].id > midnight.records[1].id
