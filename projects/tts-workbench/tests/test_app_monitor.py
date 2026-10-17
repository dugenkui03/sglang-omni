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
        f"    方法：Qwen3TTSTalkerDecoderLayer.forward()\n      内容：{message}\n\n"
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
        with archive.connection:
            archive.connection.execute(
                "DELETE FROM log_requests WHERE request_id='speech-a'"
            )
            archive.connection.execute(
                "UPDATE log_records SET category='system' WHERE text=?", (payload,)
            )
            archive.connection.execute("PRAGMA user_version=0")
    with closing(LogArchive(tmp_path, path)) as archive:
        assert archive.tail(None, "speech-a", 100).text == payload
        assert (
            archive.tail(None, None, 100, "inference", "preprocess").matched_lines == 6
        )
