"""Verify request attribution without requiring CUDA or model weights."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import StringIO
from threading import Thread

import numpy as np
import pytest
import torch
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from sglang_omni.http.request_logging import RequestLoggingApp
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.utils.logging import (
    DebugLogFormatter,
    PollingAccessFilter,
    json_for_log,
)
from sglang_omni.utils.request_context import current_request_label, request_log_context


@pytest.mark.parametrize(
    "field_name",
    ["data", "audio", "audio_path", "audio_base64", "ref_audio", "reference_audio"],
)
def test_encoded_audio_log_keeps_metadata_only(field_name: str) -> None:
    """验证嵌套参考音频即使开启预览也不会输出 Base64。"""
    encoded_audio = "UklGRkaWAABXQVZF" * 4096
    request = {"references": [{field_name: encoded_audio, "text": "参考台词"}]}
    rendered = json_for_log(request, preview_elements=5)
    summary = json.loads(rendered)
    assert encoded_audio[:32] not in rendered
    assert len(rendered) < 200
    assert summary["references"][0][field_name]["characters"] == len(encoded_audio)
    assert summary["references"][0]["text"] == "参考台词"
    assert request["references"][0][field_name] == encoded_audio


@pytest.mark.parametrize("source_kind", ["list", "numpy", "torch", "bytes"])
def test_waveform_log_omits_samples(source_kind: str) -> None:
    """验证各种音频表示仅输出元信息且不修改原数据。"""
    waveform = {
        "list": [0.125] * 24000,
        "numpy": np.full(24000, 0.125, dtype=np.float32),
        "torch": torch.full((24000,), 0.125),
        "bytes": b"sample-data" * 24000,
    }[source_kind]
    request = {"waveform": waveform, "sample_rate_hz": 24000, "duration_seconds": 1.0}
    rendered = json_for_log(request, preview_elements=5)
    summary = json.loads(rendered)
    assert "0.125" not in rendered
    assert "sample-data" not in rendered
    assert "first5" not in rendered
    assert len(rendered) < 250
    assert summary["sample_rate_hz"] == 24000
    assert summary["duration_seconds"] == 1.0
    assert request["waveform"] is waveform


def test_reference_dataclass_log_preserves_core_fields() -> None:
    """验证参考音频数据类保留核心字段且不会泄露音频数据。"""

    @dataclass
    class Reference:
        """表示参考编码器的规范化输入。"""

        ref_audio: str
        ref_text: str
        x_vector_only_mode: bool

    reference = Reference("UklGRkaWAABXQVZF" * 4096, "参考台词", False)
    summary = json.loads(json_for_log(reference))
    assert summary["ref_audio"]["characters"] == len(reference.ref_audio)
    assert summary["ref_text"] == reference.ref_text
    assert summary["x_vector_only_mode"] is False
    data_uri = "data:audio/wav;base64," + reference.ref_audio
    uri_summary = json.loads(json_for_log({"audio_path": data_uri}))["audio_path"]
    assert uri_summary["media_type"] == "audio/wav"
    assert uri_summary["characters"] == len(data_uri)


def formatted_log() -> str:
    """Exercise the production formatter using a normal, unchanged log record."""
    record = logging.LogRecord(
        "test", logging.INFO, "model.py", 312, "推理完成", (), None
    )
    return DebugLogFormatter().format(record)


@pytest.mark.parametrize("concurrency", [1, 2])
def test_simple_scheduler_restores_request_identity(concurrency: int) -> None:
    """验证串行与并发计算线程中的请求 ID 不串线。"""

    def compute(payload: str) -> str:
        """从实际调度的计算函数中读取请求日志上下文。"""
        return f"{payload}:{current_request_label()}"

    scheduler = SimpleScheduler(compute, max_concurrency=concurrency)
    thread = Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        for identifier in ("speech-one", "speech-two"):
            scheduler.inbox.put(IncomingMessage(identifier, "new_request", identifier))
        results = {scheduler.outbox.get(timeout=5).data for _ in range(2)}
        assert results == {
            "speech-one:request_id=speech-one",
            "speech-two:request_id=speech-two",
        }
        assert current_request_label() == "request_id=-"
    finally:
        scheduler.stop()
        thread.join(timeout=5)


def test_simple_scheduler_batch_lists_all_request_ids() -> None:
    """验证批量计算列出全部请求 ID，避免错误归属某个请求。"""

    def compute(payload: str) -> str:
        """提供调度器必需的单请求计算入口。"""
        return current_request_label()

    def compute_batch(payloads: list[str]) -> list[str]:
        """读取批量计算共享的日志上下文并返回每个请求的结果。"""
        return [current_request_label() for _ in payloads]

    scheduler = SimpleScheduler(compute, batch_compute_fn=compute_batch)
    loop = asyncio.new_event_loop()
    try:
        scheduler.run_batch(
            [
                IncomingMessage(identifier, "new_request", identifier)
                for identifier in ("speech-one", "speech-two")
            ],
            loop,
        )
        assert (
            scheduler.outbox.get_nowait().data == "request_ids=[speech-one, speech-two]"
        )
        assert current_request_label() == "request_id=-"
    finally:
        loop.close()


@pytest.mark.parametrize(
    ("method", "path", "status_code", "visible"),
    [
        ("GET", "/health", 200, False),
        ("GET", "/v1/models?refresh=1", 200, False),
        ("GET", "/health", 503, True),
        ("GET", "/v1/models", 500, True),
        ("POST", "/v1/audio/speech", 200, True),
        ("GET", "/other", 200, True),
    ],
)
def test_polling_access_filter(
    method: str, path: str, status_code: int, visible: bool
) -> None:
    """Quiet successful polling while retaining failed checks and generation requests."""
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "h11_impl.py",
        477,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1234", method, path, "1.1", status_code),
        None,
    )
    assert PollingAccessFilter().filter(record) is visible


def test_request_line_batch_and_exception_cleanup() -> None:
    """Put identities on the second line and restore nested and failed contexts."""
    rendered = formatted_log().splitlines()
    assert rendered[1] == "  request_id=-"
    assert rendered[3].startswith("    method：")
    assert rendered[4] == "      msg：推理完成"
    with request_log_context("speech-parent"):
        assert formatted_log().splitlines()[1] == "  request_id=speech-parent"
        with (
            pytest.raises(ValueError),
            request_log_context("speech-a", "speech-b", "speech-a"),
        ):
            assert (
                formatted_log().splitlines()[1] == "  request_ids=[speech-a, speech-b]"
            )
            raise ValueError("synthetic failure")
        assert current_request_label() == "request_id=speech-parent"
    assert current_request_label() == "request_id=-"


def test_async_and_thread_request_isolation() -> None:
    """Interleave request tasks and their offloaded work without sharing identities."""

    async def request(identifier: str) -> str:
        """Yield across another request before checking this task and its child thread."""
        with request_log_context(identifier):
            await asyncio.sleep(0)
            label = await asyncio.to_thread(current_request_label)
            assert label == f"request_id={identifier}"
            return formatted_log().splitlines()[1]

    async def concurrent_requests() -> list[str]:
        """Run overlapping synthetic identities on the same event loop."""
        return await asyncio.gather(request("speech-a"), request("speech-b"))

    assert asyncio.run(concurrent_requests()) == [
        "  request_id=speech-a",
        "  request_id=speech-b",
    ]
    assert current_request_label() == "request_id=-"


def test_scheduler_worker_rebinds_and_clears_identity() -> None:
    """Reuse a worker thread for different messages and leave idle logs unassigned."""

    def compute(payload: str) -> str:
        """Read the context deep inside an unchanged compute function."""
        assert current_request_label() == f"request_id={payload}"
        return formatted_log().splitlines()[1]

    scheduler = ThreadedSimpleScheduler(compute, max_concurrency=1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert (
            executor.submit(scheduler.run_one, "speech-a", "speech-a").result()
            == "  request_id=speech-a"
        )
        assert (
            executor.submit(scheduler.run_one, "speech-b", "speech-b").result()
            == "  request_id=speech-b"
        )
        assert executor.submit(current_request_label).result() == "request_id=-"
    scheduler.executor.shutdown()


def test_threaded_scheduler_logs_receive_and_failure_with_request_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """验证实际接收和失败回调日志均在第二行保留请求 ID。"""

    def compute(payload: str) -> str:
        """触发计算失败，覆盖线程池的异常完成回调。"""
        raise ValueError(payload)

    output = StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(DebugLogFormatter())
    logger = logging.getLogger("sglang_omni.scheduling.threaded_simple_scheduler")
    scheduler = ThreadedSimpleScheduler(compute, max_concurrency=1)
    thread = Thread(target=scheduler.start, daemon=True)
    logger.addHandler(handler)
    try:
        with caplog.at_level(logging.INFO, logger=logger.name):
            thread.start()
            scheduler.enqueue(
                IncomingMessage("speech-failed", "new_request", "failed")
            )
            assert scheduler.outbox.get(timeout=5).type == "error"
        rendered = output.getvalue()
        assert "收到新请求" in rendered
        assert "计算函数执行失败" in rendered
        assert rendered.count("  request_id=speech-failed\n") == 2
        assert "request_id=-" not in rendered
    finally:
        scheduler.stop()
        thread.join(timeout=5)
        logger.removeHandler(handler)


@pytest.mark.parametrize(
    ("method", "path", "status_code", "prefix"),
    [
        ("GET", "/health", 200, "http-"),
        ("POST", "/v1/audio/speech", 200, "speech-"),
        ("POST", "/v1/audio/speech/batch", 200, "speech-batch-"),
        ("GET", "/missing", 404, "http-"),
        ("GET", "/broken", 500, "http-"),
    ],
)
def test_http_response_and_access_logs_share_request_id(
    method: str,
    path: str,
    status_code: int,
    prefix: str,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep identity at the server send boundary, after routes and on error replies."""
    app = RequestLoggingApp()
    logger = logging.getLogger("test.http_request")
    caplog.set_level(logging.INFO, logger=logger.name)
    monkeypatch.setattr(caplog.handler, "formatter", DebugLogFormatter())

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        """Ensure a non-generation route can see its own HTTP identity."""
        logger.info("Health check")
        return JSONResponse({"request_id": request.state.request_id})

    @app.post("/v1/audio/speech")
    @app.post("/v1/audio/speech/batch")
    async def speech() -> StreamingResponse:
        """Emit logs after returning the response object, like streaming inference."""

        async def audio_chunks() -> AsyncIterator[bytes]:
            """Retain identity while awaiting and sending each response chunk."""
            for _ in range(2):
                await asyncio.sleep(0)
                logger.info("Audio chunk")
                yield b"audio"

        return StreamingResponse(audio_chunks(), media_type="audio/pcm")

    @app.get("/broken")
    async def broken() -> None:
        """Exercise FastAPI's outer 500 response, outside regular middleware."""
        raise RuntimeError("synthetic failure")

    async def serve(scope: Scope, receive: Receive, send: Send) -> None:
        """Observe the same ASGI response boundary at which Uvicorn logs access."""

        async def server_send(message: Message) -> None:
            """Log response headers only once, just as the HTTP server does."""
            if message["type"] == "http.response.start":
                logger.info(f"Access {scope['method']} {scope['path']}")
            await send(message)

        await app(scope, receive, server_send)

    with TestClient(serve, raise_server_exceptions=False) as client:
        response = client.request(method, path)
    request_id = response.headers["X-Request-Id"]
    assert response.status_code == status_code
    assert request_id.startswith(prefix)
    assert f"Access {method} {path}" in caplog.text
    labels = [
        line.strip() for line in caplog.text.splitlines() if "request_id=" in line
    ]
    assert labels and set(labels) == {f"request_id={request_id}"}
    assert current_request_label() == "request_id=-"
