"""Verify request attribution without requiring CUDA or model weights."""

import asyncio
import logging
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from sglang_omni.http.request_logging import RequestLoggingApp
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.utils.logging import DebugLogFormatter
from sglang_omni.utils.request_context import current_request_label, request_log_context


def formatted_log() -> str:
    """Exercise the production formatter using a normal, unchanged log record."""
    record = logging.LogRecord(
        "test", logging.INFO, "model.py", 312, "Forward complete", (), None
    )
    return DebugLogFormatter().format(record)


def test_request_line_batch_and_exception_cleanup() -> None:
    """Put identities on the second line and restore nested and failed contexts."""
    assert formatted_log().splitlines()[1] == "  request_id=-"
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
