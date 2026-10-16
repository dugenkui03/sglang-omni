"""Schedule bounded HTTP workloads independently of browser connections."""

import asyncio
import logging
import math
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from tts_workbench.config import RunConfig, Settings
from tts_workbench.dataset import Dataset, DialogueLine
from tts_workbench.records import RequestRecord, RunRecord
from tts_workbench.speech import read_audio_duration, receive_audio
from tts_workbench.storage import Store

logger = logging.getLogger(__name__)


@dataclass
class ActiveRun:
    """Bound in-memory state by concurrency and the selected input set."""

    record: RunRecord
    lines: list[DialogueLine]
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    tasks: set[asyncio.Task[None]] = field(default_factory=set)
    audio_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    saved_lines: set[str] = field(default_factory=set)
    audio_bytes_saved: int = 0
    quota_reported: bool = False
    failure: str | None = None


def elapsed_slots(
    now_seconds: float, next_seconds: float, interval_seconds: float
) -> int:
    """Skip expired slots arithmetically instead of catching up in a burst."""
    return max(0, math.floor((now_seconds - next_seconds) / interval_seconds))


class Runner:
    """Own one active task, its HTTP pool, and workbench-only cleanup."""

    def __init__(
        self,
        settings: Settings,
        dataset: Dataset,
        store: Store,
        client: httpx.AsyncClient,
    ) -> None:
        """Inject the client so tests can validate contracts without model inference."""
        self.settings = settings
        self.dataset = dataset
        self.store = store
        self.client = client
        self.active: ActiveRun | None = None
        self.task: asyncio.Task[None] | None = None
        self.control_lock = asyncio.Lock()

    @property
    def active_id(self) -> str | None:
        """Identify only a still-running task for GPU persistence."""
        if self.active and self.task and not self.task.done():
            return self.active.record.id
        else:
            return None

    async def start(self, config: RunConfig) -> RunRecord:
        """Start a workload without preflighting the selected audio or text."""
        async with self.control_lock:
            if self.active_id:
                raise ValueError(
                    "A task is already running; stop it before starting another"
                )
            elif not config.model.strip():
                raise ValueError("A model ID is required")
            else:
                pass
            lines = self.dataset.select(config)
            created_at = time.time()
            run = RunRecord(
                id=uuid.uuid4().hex,
                name=config.name.strip()
                or time.strftime("%m-%d %H:%M") + " " + config.language,
                status="running",
                created_at=created_at,
                config=config,
                line_ids=[line.line_id for line in lines],
                dataset_fingerprint=self.dataset.fingerprint,
            )
            await asyncio.to_thread(self.store.create_run, run)
            self.active = ActiveRun(record=run, lines=lines)
            self.task = asyncio.create_task(
                self.dispatch(self.active), name=f"workload-{run.id}"
            )
            return run

    async def stop(self, run_id: str) -> None:
        """Stop new launches while allowing in-flight attempts to finish."""
        async with self.control_lock:
            if self.active and self.active_id == run_id:
                self.active.stop.set()
                await asyncio.to_thread(self.store.update_status, run_id, "stopping")
            else:
                await asyncio.to_thread(self.store.get_run, run_id)

    async def dispatch(self, active: ActiveRun) -> None:
        """Use monotonic deadlines, bounded concurrency, and no automatic retries."""
        config = active.record.config
        interval_seconds = 60 / config.requests_per_minute
        started_seconds = time.monotonic()
        deadline_seconds = started_seconds + config.duration_seconds
        slot = 0
        sequence = 0
        skipped = 0
        persisted_skipped = 0

        def request_finished(task: asyncio.Task[None]) -> None:
            """Consume worker exceptions and turn persistence failures into task failure."""
            active.tasks.discard(task)
            if task.cancelled():
                active.failure = "A request worker was cancelled"
                active.stop.set()
            else:
                exception = task.exception()
                if exception is not None:
                    active.failure = str(exception)
                    active.stop.set()
                    logger.error(f"Request worker failed: {exception}")
                else:
                    pass

        try:
            while not active.stop.is_set():
                next_seconds = started_seconds + slot * interval_seconds
                if config.mode == "duration" and next_seconds >= deadline_seconds:
                    break
                elif config.mode != "duration" and sequence >= len(active.lines):
                    break
                else:
                    pass
                delay_seconds = next_seconds - time.monotonic()
                if delay_seconds > 0:
                    try:
                        await asyncio.wait_for(
                            active.stop.wait(), timeout=delay_seconds
                        )
                    except asyncio.TimeoutError:
                        pass
                else:
                    pass
                now_seconds = time.monotonic()
                if active.stop.is_set() or (
                    config.mode == "duration" and now_seconds >= deadline_seconds
                ):
                    break
                else:
                    pass
                missed = elapsed_slots(now_seconds, next_seconds, interval_seconds)
                slot += missed
                skipped += missed
                next_seconds = started_seconds + slot * interval_seconds
                if len(active.tasks) >= config.max_concurrent_requests:
                    skipped += 1
                else:
                    line = active.lines[sequence % len(active.lines)]
                    sequence += 1
                    task = asyncio.create_task(
                        self.request(active, line, sequence, next_seconds)
                    )
                    active.tasks.add(task)
                    task.add_done_callback(request_finished)
                slot += 1
                if skipped > persisted_skipped:
                    await asyncio.to_thread(
                        self.store.add_skipped,
                        active.record.id,
                        skipped - persisted_skipped,
                    )
                    persisted_skipped = skipped
                else:
                    pass
            if config.mode == "duration" and not active.stop.is_set():
                remaining_slots = max(
                    0, math.ceil(config.duration_seconds / interval_seconds) - slot
                )
                if remaining_slots:
                    await asyncio.to_thread(
                        self.store.add_skipped, active.record.id, remaining_slots
                    )
                else:
                    pass
            else:
                pass
            if active.tasks:
                await asyncio.gather(*list(active.tasks), return_exceptions=True)
            else:
                pass
            if active.failure:
                await asyncio.to_thread(
                    self.store.update_status, active.record.id, "failed", active.failure
                )
            else:
                status = "stopped" if active.stop.is_set() else "completed"
                await asyncio.to_thread(
                    self.store.update_status, active.record.id, status
                )
        except (OSError, sqlite3.Error, ValueError) as exception:
            logger.exception(f"Task storage or scheduling failed: {exception}")
            active.stop.set()
            active.failure = str(exception)
            if active.tasks:
                await asyncio.gather(*list(active.tasks), return_exceptions=True)
            else:
                pass
            try:
                await asyncio.to_thread(
                    self.store.update_status, active.record.id, "failed", str(exception)
                )
            except (OSError, sqlite3.Error):
                logger.exception("Could not persist failed task status")

    async def request(
        self,
        active: ActiveRun,
        line: DialogueLine,
        sequence: int,
        planned_seconds: float,
    ) -> None:
        """Keep server failures visible while sending each source line unchanged."""
        config = active.record.config
        text = line.text_zh if config.language == "Chinese" else line.text_en_tts
        record = RequestRecord(
            id=uuid.uuid4().hex,
            run_id=active.record.id,
            sequence=sequence,
            line_id=line.line_id,
            speaker=line.speaker,
            text=text,
            reference_text_needs_review=line.reference_text_needs_review,
            started_at=time.time(),
            scheduling_delay_ms=max(0, (time.monotonic() - planned_seconds) * 1000),
        )
        await asyncio.to_thread(self.store.save_request, record)
        started_seconds = time.monotonic()
        try:
            reference_uri = await asyncio.to_thread(
                self.dataset.reference_uri, line.line_id
            )
            payload = {
                "model": config.model,
                "input": text,
                "language": config.language,
                "response_format": "wav",
                "stream": False,
                "references": [{"audio_path": reference_uri, "text": line.text_zh}],
            }
            response = await asyncio.wait_for(
                receive_audio(self.client, self.settings.api_base, payload),
                timeout=config.request_timeout_seconds,
            )
            record.latency_ms = response.latency_ms
            record.http_status = response.status_code
            record.service_request_id = response.request_id
            record.finish_reason = response.finish_reason
            record.audio_bytes = len(response.body)
            if not 200 <= response.status_code < 300:
                record.status = "failed"
                record.error_type = f"http_{response.status_code}"
                record.error = response.body.decode("utf-8", errors="replace")[:4000]
            else:
                if response.finish_reason == "length":
                    record.status = "truncated"
                    record.error_type = "generation_truncated"
                    record.error = "Generation reached its token limit; the utterance may be incomplete"
                elif response.finish_reason in {
                    "error",
                    "abort",
                    "aborted",
                    "cancelled",
                }:
                    record.status = "failed"
                    record.error_type = "generation_failed"
                    record.error = f"Server finish reason: {response.finish_reason}"
                else:
                    record.status = "success"
                    record.audio_duration_seconds = await asyncio.to_thread(
                        read_audio_duration, response.body
                    )
                    await self.sample_audio(active, record, response.body)
        except (asyncio.TimeoutError, httpx.TimeoutException):
            record.status = "timeout"
            record.error_type = "request_timeout"
            record.error = f"Total request deadline exceeded ({config.request_timeout_seconds:g} seconds)"
        except httpx.RequestError as exception:
            record.status = "failed"
            record.error_type = "connection_error"
            record.error = str(exception)[:4000]
        except OSError as exception:
            record.status = "failed"
            record.error_type = "reference_read_error"
            record.error = str(exception)[:4000]
        except ValueError as exception:
            record.status = "failed"
            record.error_type = "client_error"
            record.error = str(exception)[:4000]
        finally:
            record.finished_at = time.time()
            if record.latency_ms is None:
                record.latency_ms = (time.monotonic() - started_seconds) * 1000
            else:
                pass
            await asyncio.to_thread(self.store.save_request, record)

    async def sample_audio(
        self, active: ActiveRun, record: RequestRecord, audio_bytes: bytes
    ) -> None:
        """Save at most one successful sample per selected line within the quota."""
        async with active.audio_lock:
            if record.line_id in active.saved_lines:
                return
            elif (
                active.audio_bytes_saved + len(audio_bytes)
                > self.settings.audio_quota_bytes
            ):
                if not active.quota_reported:
                    active.quota_reported = True
                    await asyncio.to_thread(
                        self.store.event,
                        record.run_id,
                        "warning",
                        "Audio quota reached; request accounting continues",
                    )
                else:
                    pass
                return
            else:
                pass
            destination = self.store.audio_root / record.run_id / f"{record.id}.wav"
            try:
                await asyncio.to_thread(self.write_audio, destination, audio_bytes)
                record.audio_path = str(destination.relative_to(self.store.root))
                active.saved_lines.add(record.line_id)
                active.audio_bytes_saved += len(audio_bytes)
                await asyncio.to_thread(
                    self.store.add_audio_bytes, record.run_id, len(audio_bytes)
                )
            except OSError as exception:
                active.failure = f"Audio storage failed: {exception}"
                active.stop.set()
                await asyncio.to_thread(
                    self.store.event, record.run_id, "error", active.failure
                )

    def write_audio(self, destination: Path, audio_bytes: bytes) -> None:
        """Publish a sampled file atomically so readers never receive partial WAVs."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".partial")
        temporary.write_bytes(audio_bytes)
        temporary.replace(destination)

    async def cleanup(self, run_id: str | None) -> int:
        """Delete only terminal workbench tasks, serializing against new starts."""
        async with self.control_lock:
            identifiers = (
                [run_id]
                if run_id
                else await asyncio.to_thread(self.store.completed_ids)
            )
            for identifier in identifiers:
                run = await asyncio.to_thread(self.store.get_run, identifier)
                if (
                    run.status in {"running", "stopping"}
                    or identifier == self.active_id
                ):
                    raise ValueError("A running task cannot be cleared")
                else:
                    directory = self.store.audio_root / identifier
                    if (
                        directory.is_symlink()
                        or directory.resolve().parent != self.store.audio_root
                    ):
                        raise ValueError("Unexpected audio result path")
                    elif directory.exists():
                        await asyncio.to_thread(shutil.rmtree, directory)
                    else:
                        pass
                    await asyncio.to_thread(self.store.delete_run, identifier)
            return len(identifiers)

    async def shutdown(self) -> None:
        """Drain existing attempts before closing the shared client and database."""
        if self.active and self.task and not self.task.done():
            self.active.stop.set()
            await self.task
        else:
            pass
