"""Observe local GPU devices and a configured log without managing TTS."""

import asyncio
import codecs
import csv
import hashlib
import io
import logging
import sqlite3
import time
from collections import deque
from pathlib import Path

import httpx
from pydantic import BaseModel, ValidationError

from tts_workbench.records import GpuSample
from tts_workbench.runner import Runner
from tts_workbench.storage import Store

logger = logging.getLogger(__name__)
GPU_INTERVAL_SECONDS = 2
MAX_LOG_BYTES = 8 * 1024 * 1024
MAX_LOG_LINES = 20000


class GpuState(BaseModel):
    """Expose unavailable telemetry explicitly rather than substituting zeros."""

    sampled_at: float | None = None
    devices: list[GpuSample] = []
    error: str | None = "GPU sampling has not started"


class GpuView(BaseModel):
    """Combine the current device status with a bounded chart history."""

    current: GpuState
    history: list[GpuSample]


class LogTail(BaseModel):
    """Return a byte cursor with replacement and truncation detection."""

    source: str | None
    cursor: str | None = None
    text: str = ""
    reset: bool = False
    truncated: bool = False
    error: str | None = None


class ModelCard(BaseModel):
    """Read the standard model identifier from an HTTP service."""

    id: str


class ModelList(BaseModel):
    """Validate the OpenAI-compatible model list at the HTTP boundary."""

    data: list[ModelCard]


class HealthPayload(BaseModel):
    """Read the health field used by sglang-omni."""

    status: str = "unknown"


class ServiceState(BaseModel):
    """Keep connectivity checks separate from workload request statistics."""

    api_base: str
    checked_at: float
    healthy: bool
    models: list[str]
    error: str | None = None


class ServiceMonitor:
    """Cache low-frequency health and model queries to the configured target."""

    def __init__(self, api_base: str, client: httpx.AsyncClient) -> None:
        """Share the authorized HTTP client without exposing its credentials."""
        self.api_base = api_base.rstrip("/")
        self.client = client
        self.lock = asyncio.Lock()
        self.cached: ServiceState | None = None

    async def get(self) -> ServiceState:
        """Report a failed health check without changing or stopping the service."""
        async with self.lock:
            if self.cached and time.time() - self.cached.checked_at < 5:
                return self.cached
            else:
                pass
            models = self.cached.models if self.cached else []
            healthy = False
            error = None
            try:
                health_response, model_response = await asyncio.wait_for(
                    asyncio.gather(
                        self.client.get(f"{self.api_base}/health"),
                        self.client.get(f"{self.api_base}/v1/models"),
                    ),
                    timeout=5,
                )
                if health_response.is_success:
                    health = HealthPayload.model_validate_json(health_response.content)
                    healthy = health.status == "healthy"
                    error = None if healthy else f"Service state: {health.status}"
                else:
                    error = (
                        f"Health endpoint returned HTTP {health_response.status_code}"
                    )
                if model_response.is_success:
                    models = [
                        model.id
                        for model in ModelList.model_validate_json(
                            model_response.content
                        ).data
                    ]
                else:
                    error = (
                        error
                        or f"Model endpoint returned HTTP {model_response.status_code}"
                    )
            except (
                httpx.HTTPError,
                asyncio.TimeoutError,
                ValidationError,
            ) as exception:
                error = f"Service connection unavailable: {type(exception).__name__}: {exception}"
            self.cached = ServiceState(
                api_base=self.api_base,
                checked_at=time.time(),
                healthy=healthy,
                models=models,
                error=error,
            )
            return self.cached


def optional_number(value: str) -> float | None:
    """Preserve unsupported NVIDIA fields as null instead of zero."""
    try:
        return float(value.strip())
    except ValueError:
        return None


class GpuMonitor:
    """Sample every visible NVIDIA device without loading CUDA libraries."""

    def __init__(self, store: Store, runner: Runner) -> None:
        """Bound live history to 30 minutes while persisting active-run history."""
        self.store = store
        self.runner = runner
        self.current = GpuState()
        self.history: deque[list[GpuSample]] = deque(maxlen=900)
        self.stop = asyncio.Event()

    async def sample(self) -> GpuState:
        """Query NVIDIA's CLI with a total subprocess timeout."""
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            output, errors = await asyncio.wait_for(process.communicate(), timeout=5)
            if process.returncode:
                return GpuState(
                    sampled_at=time.time(),
                    error=errors.decode(errors="replace")[:1000] or "nvidia-smi failed",
                )
            else:
                devices: list[GpuSample] = []
                timestamp = time.time()
                for row in csv.reader(io.StringIO(output.decode())):
                    if len(row) != 8:
                        raise ValueError("Unexpected nvidia-smi column count")
                    else:
                        devices.append(
                            GpuSample(
                                timestamp=timestamp,
                                index=row[0].strip(),
                                uuid=row[1].strip(),
                                name=row[2].strip(),
                                memory_used_mib=optional_number(row[3]),
                                memory_total_mib=optional_number(row[4]),
                                utilization_percent=optional_number(row[5]),
                                temperature_celsius=optional_number(row[6]),
                                power_watts=optional_number(row[7]),
                            )
                        )
                return GpuState(
                    sampled_at=timestamp,
                    devices=devices,
                    error=None if devices else "No NVIDIA GPU was reported",
                )
        except asyncio.TimeoutError:
            if process is not None:
                process.kill()
                await process.communicate()
            else:
                pass
            return GpuState(
                sampled_at=time.time(), error="nvidia-smi exceeded its 5 second timeout"
            )
        except (OSError, ValueError) as exception:
            return GpuState(
                sampled_at=time.time(), error=f"GPU telemetry unavailable: {exception}"
            )

    async def run(self) -> None:
        """Persist samples off the event loop without stopping the request scheduler."""
        while not self.stop.is_set():
            self.current = await self.sample()
            if self.current.devices:
                self.history.append(self.current.devices)
                run_id = self.runner.active_id
                if run_id:
                    try:
                        await asyncio.to_thread(
                            self.store.save_gpu, run_id, self.current.devices
                        )
                    except sqlite3.Error as exception:
                        self.current.error = (
                            f"GPU history could not be saved: {exception}"
                        )
                        logger.error(self.current.error)
                else:
                    pass
            else:
                pass
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=GPU_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                pass


def tail_log(
    path: Path | None, cursor: str | None, limit: int = MAX_LOG_LINES
) -> LogTail:
    """Follow only the configured file and handle rotation or truncate/regrow."""
    if path is None:
        return LogTail(source=None, error="No TTS log file was configured")
    else:
        pass
    try:
        with path.open("rb") as stream:
            metadata = path.stat()
            identity = f"{metadata.st_dev}:{metadata.st_ino}"
            offset = max(0, metadata.st_size - MAX_LOG_BYTES)
            reset = True
            if cursor:
                parts = cursor.split(":")
                if len(parts) != 4:
                    raise ValueError("Invalid log cursor")
                else:
                    previous_offset = int(parts[2])
                    if (
                        ":".join(parts[:2]) == identity
                        and 0 <= previous_offset <= metadata.st_size
                    ):
                        stream.seek(max(0, previous_offset - 32))
                        prefix = stream.read(min(32, previous_offset))
                        if hashlib.sha256(prefix).hexdigest()[:16] == parts[3]:
                            offset = previous_offset
                            reset = False
                        else:
                            pass
                    else:
                        pass
            else:
                pass
            if metadata.st_size - offset > 2 * MAX_LOG_BYTES:
                offset = max(0, metadata.st_size - MAX_LOG_BYTES)
                reset = True
            else:
                pass
            stream.seek(offset)
            content = stream.read(MAX_LOG_BYTES)
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            text = decoder.decode(content, final=False)
            remaining_bytes, _ = decoder.getstate()
            end_offset = offset + len(content) - len(remaining_bytes)
            if reset and offset > 0:
                _, separator, suffix = text.partition("\n")
                if separator:
                    text = suffix
                else:
                    pass
            else:
                pass
            lines = text.splitlines(keepends=True)
            if len(lines) > limit:
                text = "".join(lines[-limit:])
                reset = True
            else:
                pass
            stream.seek(max(0, end_offset - 32))
            fingerprint = hashlib.sha256(stream.read(min(32, end_offset))).hexdigest()[
                :16
            ]
            return LogTail(
                source=path.name,
                cursor=f"{identity}:{end_offset}:{fingerprint}",
                text=text,
                reset=reset,
            )
    except (OSError, ValueError) as exception:
        return LogTail(source=path.name, error=f"Log file unavailable: {exception}")
