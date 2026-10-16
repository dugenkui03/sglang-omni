"""Verify authentication, data boundaries, and log cursor behavior."""

import asyncio
import base64
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tts_workbench.app import create_app
from tts_workbench.config import Settings
from tts_workbench.monitor import MAX_LOG_BYTES, optional_number, tail_log


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
