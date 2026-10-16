"""Verify request attribution without requiring CUDA or model weights."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

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
