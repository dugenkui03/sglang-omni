"""Verify observable scheduling, audio, persistence, and cleanup contracts."""

import asyncio
import base64
import io
import json
import time
import wave
from dataclasses import replace

import httpx
import pytest

from tts_workbench.config import RunConfig, Settings
from tts_workbench.dataset import Dataset
from tts_workbench.records import RequestRecord, RunRecord
from tts_workbench.runner import Runner, elapsed_slots
from tts_workbench.storage import Store


@pytest.mark.parametrize("language", ["Chinese", "English"])
async def test_complete_unit_payload_and_cleanup(
    settings: Settings, wav_bytes: bytes, language: str
) -> None:
    """Send every line in order with Chinese references, then clear only owned output."""
    observed: list[dict[str, object]] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        """Capture wire payloads while returning a clearly synthetic test WAV."""
        observed.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=wav_bytes,
            headers={"X-Finish-Reason": "stop", "X-Request-Id": "test-only"},
        )

    store = Store(settings.results_directory)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        runner = Runner(settings, Dataset(settings.data_directory), store, client)
        run = await runner.start(
            RunConfig(
                model="test-only",
                mode="once",
                language=language,
                requests_per_minute=6000,
            )
        )
        await runner.task
        assert store.get_run(run.id).status == "completed"
        assert store.statistics(run.id).success == 2
        assert [payload["input"] for payload in observed] == (
            ["中文台词0", "中文台词1"]
            if language == "Chinese"
            else ["English line 0", "English line 1"]
        )
        assert observed[0]["references"][0]["text"] == "中文台词0"
        assert (
            base64.b64decode(observed[0]["references"][0]["audio_path"].split(",")[1])
            == wav_bytes
        )
        assert len(list(store.audio_root.rglob("*.wav"))) == 2
        await runner.cleanup(run.id)
        assert not store.list_runs()
        assert (settings.data_directory / "reference.wav").read_bytes() == wav_bytes
        assert settings.log_file.read_text() == "original service log\n"
    store.close()


async def test_duration_concurrency_skip_stop_and_sampling(
    settings: Settings, wav_bytes: bytes
) -> None:
    """Saturate a single worker and verify no queue, retry, or duplicate sample appears."""
    simultaneous = 0
    maximum = 0
    starts: list[float] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        """Model a slow HTTP response, independent of any actual GPU service."""
        nonlocal simultaneous, maximum
        assert request.url.path == "/v1/audio/speech"
        simultaneous += 1
        maximum = max(maximum, simultaneous)
        starts.append(time.monotonic())
        await asyncio.sleep(0.055)
        simultaneous -= 1
        return httpx.Response(
            200, content=wav_bytes, headers={"X-Finish-Reason": "stop"}
        )

    store = Store(settings.results_directory)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        runner = Runner(settings, Dataset(settings.data_directory), store, client)
        run = await runner.start(
            RunConfig(
                model="test-only",
                duration_seconds=5,
                requests_per_minute=6000,
                max_concurrent_requests=1,
            )
        )
        await asyncio.sleep(0.24)
        with pytest.raises(ValueError, match="running"):
            await runner.cleanup(run.id)
        with pytest.raises(ValueError, match="already running"):
            await runner.start(RunConfig(model="test-only"))
        await runner.stop(run.id)
        await runner.task
        final = store.get_run(run.id)
        assert final.status == "stopped" and final.skipped_count > 0
        assert maximum == 1 and 2 <= len(starts) <= 5
        assert all(
            later - earlier >= 0.05 for earlier, later in zip(starts, starts[1:])
        )
        assert len(list(store.audio_root.rglob("*.wav"))) == 2
        assert store.statistics(run.id).sent == len(starts)
    store.close()


async def test_duration_ends_without_browser(
    settings: Settings, wav_bytes: bytes
) -> None:
    """Complete a timed run using backend ownership alone."""

    async def transport(request: httpx.Request) -> httpx.Response:
        """Return immediate synthetic audio for deterministic pacing checks."""
        assert request.method == "POST"
        return httpx.Response(200, content=wav_bytes)

    store = Store(settings.results_directory)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        runner = Runner(settings, Dataset(settings.data_directory), store, client)
        run = await runner.start(
            RunConfig(model="test-only", requests_per_minute=600, duration_seconds=0.25)
        )
        await runner.task
        records = store.request_page(run.id, 0, 30, False).records
        assert store.get_run(run.id).status == "completed"
        assert 2 <= len(records) <= 3
        assert all(record.started_at - run.created_at < 0.3 for record in records)
    store.close()


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("http429", "failed"),
        ("http500", "failed"),
        ("invalid", "success"),
        ("error", "failed"),
        ("length", "truncated"),
        ("timeout", "timeout"),
        ("unknown", "success"),
    ],
)
async def test_failure_classification(
    settings: Settings, wav_bytes: bytes, scenario: str, expected: str
) -> None:
    """Use server outcomes without rejecting a successful response's audio content."""

    async def transport(request: httpx.Request) -> httpx.Response:
        """Supply the selected failure at the external HTTP boundary."""
        assert request.url.host == "tts.test"
        if scenario.startswith("http"):
            return httpx.Response(
                int(scenario[4:]), json={"error": "synthetic-test-only"}
            )
        elif scenario == "timeout":
            await asyncio.sleep(0.1)
            return httpx.Response(200, content=wav_bytes)
        else:
            return httpx.Response(
                200,
                content=b"not audio" if scenario == "invalid" else wav_bytes,
                headers={
                    "X-Finish-Reason": (
                        scenario
                        if scenario in {"length", "unknown", "error"}
                        else "stop"
                    )
                },
            )

    store = Store(settings.results_directory)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        runner = Runner(settings, Dataset(settings.data_directory), store, client)
        run = await runner.start(
            RunConfig(
                model="test-only",
                mode="single",
                line_ids=["unit_0"],
                request_timeout_seconds=0.02,
            )
        )
        await runner.task
        page = store.request_page(run.id, 0, 30, False)
        assert page.total == 1 and page.records[0].status == expected
        statistics = store.statistics(run.id)
        assert statistics.failed == (0 if expected == "success" else 1)
        assert statistics.unknown_finish == (1 if scenario == "unknown" else 0)
        if expected != "success":
            assert statistics.average_latency_ms is None
            assert statistics.p95_latency_ms is None
            assert statistics.successful_requests_per_minute == 0
            assert statistics.successful_audio_seconds == 0
            assert statistics.measured_audio_count == 0
        elif scenario == "invalid":
            assert page.records[0].audio_duration_seconds is None
            assert statistics.measured_audio_count == 0
            assert statistics.success == 1
            assert (
                store.root / page.records[0].audio_path
            ).read_bytes() == b"not audio"
        else:
            assert statistics.measured_audio_count == 1
    store.close()


async def test_audio_quota_preserves_all_accounting(
    settings: Settings, wav_bytes: bytes
) -> None:
    """Exhaust audio retention without converting successful requests to failures."""

    async def transport(request: httpx.Request) -> httpx.Response:
        """Return the same valid fixture for both selected dialogue lines."""
        assert request.method == "POST"
        return httpx.Response(
            200, content=wav_bytes, headers={"X-Finish-Reason": "stop"}
        )

    store = Store(settings.results_directory)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        runner = Runner(
            replace(settings, audio_quota_bytes=1),
            Dataset(settings.data_directory),
            store,
            client,
        )
        run = await runner.start(
            RunConfig(model="test-only", mode="once", requests_per_minute=6000)
        )
        await runner.task
        assert store.statistics(run.id).success == 2
        assert not list(store.audio_root.rglob("*.wav"))
        assert sum("quota" in event.message for event in store.events(run.id)) == 1
    store.close()


def test_restart_recovery_and_percentiles(settings: Settings) -> None:
    """Retain abandoned attempts and compute nearest-rank latency percentiles."""
    store = Store(settings.results_directory)
    run = RunRecord(
        id="restart-test",
        name="test",
        status="running",
        created_at=time.time(),
        config=RunConfig(model="test"),
        line_ids=["unit_0"],
        dataset_fingerprint="test",
    )
    store.create_run(run)
    for sequence, milliseconds in enumerate((10, 20, 30), start=1):
        store.save_request(
            RequestRecord(
                id=f"request-{sequence}",
                run_id=run.id,
                sequence=sequence,
                line_id="unit_0",
                speaker="test",
                text="test",
                reference_text_needs_review=False,
                started_at=time.time(),
                status="success",
                latency_ms=milliseconds,
                finish_reason="stop",
                finished_at=time.time(),
                audio_duration_seconds=0.2,
            )
        )
    for sequence, status in enumerate(("failed", "timeout", "truncated"), start=4):
        store.save_request(
            RequestRecord(
                id=f"request-{sequence}",
                run_id=run.id,
                sequence=sequence,
                line_id="unit_0",
                speaker="test",
                text="test",
                reference_text_needs_review=False,
                started_at=time.time(),
                finished_at=time.time(),
                status=status,
                latency_ms=10000,
                audio_duration_seconds=999,
            )
        )
    store.save_request(
        RequestRecord(
            id="unfinished",
            run_id=run.id,
            sequence=7,
            line_id="unit_0",
            speaker="test",
            text="test",
            reference_text_needs_review=False,
            started_at=time.time(),
        )
    )
    store.close()
    recovered = Store(settings.results_directory)
    recovered.recover_interrupted()
    assert recovered.get_run(run.id).status == "interrupted"
    statistics = recovered.statistics(run.id)
    assert statistics.interrupted == 1 and statistics.sent == 7
    assert statistics.failed == 4 and statistics.success == 3
    assert statistics.average_latency_ms == 20
    assert statistics.p50_latency_ms == 20 and statistics.p99_latency_ms == 30
    assert statistics.successful_requests_per_minute == 180
    assert statistics.successful_audio_seconds == pytest.approx(0.6)
    assert statistics.measured_audio_count == 3
    assert recovered.timeline(run.id)[0].average_latency_ms == 20
    recovered.close()


def test_elapsed_slots_handles_day_long_gaps_without_queue() -> None:
    """Account for missed launches in constant space even after a long pause."""
    assert elapsed_slots(86400, 0, 6) == 14400
    assert elapsed_slots(5.99, 6, 6) == 0


@pytest.mark.parametrize("readable_audio", [True, False])
async def test_unfiltered_reference_and_response(
    settings: Settings, readable_audio: bool
) -> None:
    """Forward short or undecodable bytes and empty text without a checksum gate."""
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(bytes(48))
    reference_bytes = output.getvalue() if readable_audio else b"original bad audio"
    (settings.data_directory / "reference.wav").write_bytes(reference_bytes)
    lines_path = settings.data_directory / "lines.jsonl"
    lines = [json.loads(row) for row in lines_path.read_text().splitlines()]
    lines[0]["text_zh"] = ""
    lines_path.write_text("\n".join(json.dumps(line) for line in lines))
    observed: list[dict[str, object]] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        """Observe the exact payload and return the same bytes without evaluation."""
        observed.append(json.loads(request.content))
        return httpx.Response(200, content=reference_bytes)

    store = Store(settings.results_directory)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        runner = Runner(settings, Dataset(settings.data_directory), store, client)
        run = await runner.start(
            RunConfig(model="test-only", mode="once", requests_per_minute=6000)
        )
        await runner.task
        assert len(observed) == 2
        assert observed[0]["input"] == ""
        assert observed[0]["references"][0]["text"] == ""
        assert (
            base64.b64decode(observed[0]["references"][0]["audio_path"].split(",")[1])
            == reference_bytes
        )
        statistics = store.statistics(run.id)
        assert statistics.success == 2 and statistics.failed == 0
        assert statistics.measured_audio_count == (2 if readable_audio else 0)
        assert statistics.successful_audio_seconds == pytest.approx(
            0.002 if readable_audio else 0
        )
    store.close()


async def test_missing_reference_is_recorded_per_line(
    settings: Settings, wav_bytes: bytes
) -> None:
    """Continue other requests when an individual source file cannot be read."""
    lines_path = settings.data_directory / "lines.jsonl"
    lines = [json.loads(row) for row in lines_path.read_text().splitlines()]
    lines[0]["audio_path"] = "missing.wav"
    lines_path.write_text("\n".join(json.dumps(line) for line in lines))
    observed: list[str] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        """Capture the remaining readable line at the network boundary."""
        observed.append(request.content.decode())
        return httpx.Response(200, content=wav_bytes)

    store = Store(settings.results_directory)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        runner = Runner(settings, Dataset(settings.data_directory), store, client)
        run = await runner.start(
            RunConfig(model="test-only", mode="once", requests_per_minute=6000)
        )
        await runner.task
        assert len(observed) == 1
        assert store.get_run(run.id).status == "completed"
        assert store.statistics(run.id).success == 1
        failure = store.request_page(run.id, 0, 30, True).records[0]
        assert (
            failure.line_id == "unit_0" and failure.error_type == "reference_read_error"
        )
    store.close()
