"""Verify authentication, data boundaries, and log cursor behavior."""

import asyncio
import base64
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tts_workbench.app import create_app
from tts_workbench.config import RunConfig, Settings
from tts_workbench.monitor import (
    MAX_LOG_BYTES,
    MAX_LOG_LINES,
    optional_number,
    tail_log,
)
from tts_workbench.records import RequestRecord, RunRecord
from tts_workbench.request_logs import download_request_log, tail_request_log


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
    """Read the latest 20k lines and discard a stale backlog beyond the byte window."""
    path = tmp_path / "large.log"
    path.write_text("".join(f"line {index}\n" for index in range(25000)))
    first = tail_log(path, None)
    assert len(first.text.splitlines()) == MAX_LOG_LINES
    assert first.text.startswith("line 5000\n")
    smaller = tail_log(path, None, 100)
    assert len(smaller.text.splitlines()) == 100
    assert smaller.text.startswith("line 24900\n")
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
    """Match exact batch members, retain complete records and reread incomplete writes."""
    path = tmp_path / "server.log"
    first = request_record("request_id=speech-a", "开始注意力计算，layer_id=12")
    other = request_record("request_id=speech-a-extra", "mentions speech-a")
    batch = request_record("request_ids=[speech-b, speech-a]", "合批计算")
    path.write_text(first + other + batch[:-4])
    initial = tail_request_log(path, None, "speech-a", 20000)
    assert initial.text == first
    with path.open("a") as stream:
        stream.write(batch[-4:])
    appended = tail_request_log(path, initial.cursor, "speech-a", 20000)
    assert appended.text == batch and not appended.reset
    assert tail_request_log(path, appended.cursor, "speech-a", 20000).text == ""
    smaller = tail_request_log(path, None, "speech-a", 7)
    assert smaller.text == batch and smaller.truncated
    assert b"".join(download_request_log(path, "speech-a")).decode() == first + batch
    changed_filter = tail_request_log(path, appended.cursor, "speech-a-extra", 20000)
    assert changed_filter.reset and changed_filter.text == other
    path.rename(tmp_path / "old.log")
    path.write_text(first)
    rotated = tail_request_log(path, appended.cursor, "speech-a", 20000)
    assert rotated.reset and rotated.text == first


def test_request_search_reads_history_outside_latest_byte_window(
    tmp_path: Path,
) -> None:
    """Find an old request even after unrelated logs exceed the normal tail window."""
    path = tmp_path / "large.log"
    old = request_record("request_id=speech-old", "historical request")
    recent = request_record("request_id=http-health", "health check")
    path.write_text(old + recent * (MAX_LOG_BYTES // len(recent.encode()) + 1))
    assert "speech-old" not in tail_log(path, None).text
    found = tail_request_log(path, None, "speech-old", 20000)
    assert found.text == old


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
        download = client.get(
            "/api/workbench/logs/download", params={"request_id": "speech-a"}
        )
        assert download.status_code == 200
        assert download.text == first + second
        assert "attachment" in download.headers["content-disposition"]


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
                        started_at=index * 2 + sequence,
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
