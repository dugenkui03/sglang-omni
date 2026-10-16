"""Define typed records shared by the database and observation APIs."""

from typing import Literal

from pydantic import BaseModel

from tts_workbench.config import RunConfig

RunStatus = Literal[
    "running", "stopping", "completed", "stopped", "interrupted", "failed"
]
RequestStatus = Literal[
    "inflight", "success", "failed", "timeout", "truncated", "interrupted"
]


class RunRecord(BaseModel):
    """Expose reproducible task settings and lifecycle timestamps."""

    id: str
    name: str
    status: RunStatus
    created_at: float
    ended_at: float | None = None
    config: RunConfig
    line_ids: list[str]
    dataset_fingerprint: str
    skipped_count: int = 0
    error: str | None = None
    audio_bytes_saved: int = 0


class RequestRecord(BaseModel):
    """Retain one attempt independently of whether audio was sampled."""

    id: str
    run_id: str
    sequence: int
    line_id: str
    speaker: str
    text: str
    reference_text_needs_review: bool
    status: RequestStatus = "inflight"
    started_at: float
    finished_at: float | None = None
    latency_ms: float | None = None
    scheduling_delay_ms: float = 0
    http_status: int | None = None
    service_request_id: str | None = None
    finish_reason: str | None = None
    error_type: str | None = None
    error: str | None = None
    audio_duration_seconds: float | None = None
    audio_bytes: int = 0
    audio_path: str | None = None


class RunStatistics(BaseModel):
    """Separate terminal failure categories from the overall failure count."""

    sent: int = 0
    inflight: int = 0
    success: int = 0
    failed: int = 0
    timeout: int = 0
    truncated: int = 0
    interrupted: int = 0
    unknown_finish: int = 0
    covered_lines: int = 0
    successful_lines: int = 0
    average_latency_ms: float | None = None
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    actual_requests_per_minute: float = 0
    successful_requests_per_minute: float = 0
    successful_audio_seconds: float = 0
    measured_audio_count: int = 0


class NamedRequestRecord(RequestRecord):
    """Include the owning task name in read-only request search results."""

    run_name: str


class RequestPage(BaseModel):
    """Return a bounded page of attempts."""

    total: int
    records: list[NamedRequestRecord]


class TimelinePoint(BaseModel):
    """Aggregate requests into chart-sized time buckets."""

    timestamp: float
    sent: int
    success: int
    failed: int
    average_latency_ms: float | None


class EventRecord(BaseModel):
    """Describe task lifecycle and storage events."""

    timestamp: float
    level: str
    message: str


class GpuSample(BaseModel):
    """Represent NVIDIA device-level utilization rather than SM occupancy."""

    timestamp: float
    uuid: str
    index: str
    name: str
    memory_used_mib: float | None
    memory_total_mib: float | None
    utilization_percent: float | None
    temperature_celsius: float | None
    power_watts: float | None
