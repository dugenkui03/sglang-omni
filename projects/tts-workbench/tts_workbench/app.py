"""Serve the workbench and its authenticated observation APIs."""

import asyncio
import base64
import binascii
import fcntl
import logging
import re
import secrets
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tts_workbench.config import RunConfig, Settings
from tts_workbench.dataset import DataCatalog, Dataset
from tts_workbench.log_search import LogSearchPage, search_log_history
from tts_workbench.monitor import (
    DEFAULT_LOG_LINES,
    MAX_LOG_LINES,
    GpuMonitor,
    GpuView,
    LogTail,
    ServiceMonitor,
    ServiceState,
)
from tts_workbench.records import (
    EventRecord,
    RequestPage,
    RunRecord,
    RunStatistics,
    TimelinePoint,
)
from tts_workbench.request_logs import LogArchive
from tts_workbench.runner import Runner
from tts_workbench.snapshots import HttpRequestDetails
from tts_workbench.speech import create_speech_payload
from tts_workbench.storage import Store

logger = logging.getLogger(__name__)
STATIC_DIRECTORY = Path(__file__).with_name("static")


class NextResponse(Protocol):
    """Type Starlette's HTTP middleware continuation."""

    async def __call__(self, request: Request) -> Response:
        """Dispatch the authenticated request to the next application layer."""
        ...


@dataclass
class Runtime:
    """Keep service resources scoped to the application lifespan."""

    dataset: Dataset
    store: Store
    runner: Runner
    gpu: GpuMonitor
    service: ServiceMonitor
    logs: LogArchive


class RunCollection(BaseModel):
    """Report active identity separately from the most recent task list."""

    runs: list[RunRecord]
    active_id: str | None
    runtime_error: str | None = None


class RunDetail(BaseModel):
    """Return the task summary and chart-sized observations."""

    run: RunRecord
    statistics: RunStatistics
    timeline: list[TimelinePoint]
    events: list[EventRecord]


class ClearResult(BaseModel):
    """Count tasks actually removed by a cleanup action."""

    cleared: int


def runtime_from(request: Request) -> Runtime:
    """Read resources initialized by FastAPI's lifespan."""
    return request.app.state.runtime


def create_app(settings: Settings) -> FastAPI:
    """Build an independent application without importing any inference modules."""
    settings.validate()
    dataset = Dataset(settings.data_directory)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Own a single scheduler per result directory across process restarts."""
        settings.results_directory.mkdir(parents=True, exist_ok=True)
        with (settings.results_directory / ".workbench.lock").open("a+b") as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exception:
                raise RuntimeError(
                    "Another workbench already owns this results directory"
                ) from exception
            store = Store(settings.results_directory)
            store.recover_interrupted()
            headers = (
                {"Authorization": f"Bearer {settings.api_key}"}
                if settings.api_key
                else {}
            )
            async with httpx.AsyncClient(
                headers=headers,
                timeout=None,
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=64),
                trust_env=False,
            ) as client:
                runner = Runner(settings, dataset, store, client)
                gpu = GpuMonitor(store, runner)
                logs = LogArchive(settings.results_directory, settings.log_file)
                app.state.runtime = Runtime(
                    dataset,
                    store,
                    runner,
                    gpu,
                    ServiceMonitor(settings.api_base, client),
                    logs,
                )
                gpu_task = asyncio.create_task(gpu.run(), name="gpu-observer")
                log_task = asyncio.create_task(logs.run(), name="log-archive")
                try:
                    yield
                finally:
                    gpu.stop.set()
                    logs.stop.set()
                    await asyncio.gather(gpu_task, log_task)
                    await runner.shutdown()
                    logs.close()
                    store.close()

    app = FastAPI(
        title="TTS Workbench",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )

    @app.middleware("http")
    async def authenticate(request: Request, call_next: NextResponse) -> Response:
        """Apply authentication to assets too and reject cross-origin mutations."""
        if settings.password:
            username = password = ""
            authorization = request.headers.get("authorization", "")
            if authorization.startswith("Basic "):
                try:
                    decoded = base64.b64decode(authorization[6:], validate=True).decode(
                        "utf-8"
                    )
                    username, _, password = decoded.partition(":")
                except (binascii.Error, UnicodeError):
                    pass
            else:
                pass
            if not (
                secrets.compare_digest(username.encode(), b"workbench")
                and secrets.compare_digest(
                    password.encode(), settings.password.encode()
                )
            ):
                return JSONResponse(
                    {"detail": "Workbench credentials are required"},
                    status_code=401,
                    headers={
                        "WWW-Authenticate": 'Basic realm="TTS Workbench", charset="UTF-8"'
                    },
                )
            else:
                pass
        else:
            pass
        if request.method in {"POST", "DELETE", "PUT", "PATCH"}:
            origin = request.headers.get("origin")
            if request.headers.get("X-Workbench-Client") != "1":
                return JSONResponse(
                    {"detail": "Missing X-Workbench-Client header"}, status_code=403
                )
            elif origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse(
                    {"detail": "Cross-origin changes are not allowed"}, status_code=403
                )
            else:
                pass
        else:
            pass
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(KeyError)
    async def missing_resource(request: Request, exception: KeyError) -> JSONResponse:
        """Return a stable not-found contract for unknown task or audio IDs."""
        logger.info(f"Resource not found at {request.url.path}: {exception}")
        return JSONResponse(
            {"detail": "Requested record was not found"}, status_code=404
        )

    @app.exception_handler(ValueError)
    async def invalid_operation(
        request: Request, exception: ValueError
    ) -> JSONResponse:
        """Expose actionable validation failures without mutating existing tasks."""
        logger.info(f"Rejected operation at {request.url.path}: {exception}")
        return JSONResponse({"detail": str(exception)}, status_code=400)

    @app.exception_handler(sqlite3.Error)
    async def storage_unavailable(
        request: Request, exception: sqlite3.Error
    ) -> JSONResponse:
        """Report storage failures explicitly so the UI cannot claim persistence."""
        logger.error(f"Storage failure at {request.url.path}: {exception}")
        return JSONResponse(
            {"detail": "Result storage is unavailable; check the workbench log"},
            status_code=503,
        )

    @app.get("/", response_class=FileResponse)
    async def index() -> FileResponse:
        """Serve the packaged browser application."""
        return FileResponse(STATIC_DIRECTORY / "index.html")

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        """Check workbench readiness independently of model readiness."""
        return {"status": "healthy"}

    @app.get("/api/workbench/dataset")
    async def catalog(request: Request) -> DataCatalog:
        """List the complete portable reference dataset."""
        return runtime_from(request).dataset.public_manifest()

    @app.get("/api/workbench/service")
    async def service(request: Request) -> ServiceState:
        """Read the configured target's health and model list."""
        return await runtime_from(request).service.get()

    @app.get("/api/workbench/task-name")
    async def task_name(request: Request) -> dict[str, str]:
        """Preview the fixed model's next name; creation allocates the actual sequence."""
        name = await asyncio.to_thread(
            runtime_from(request).store.next_run_name, RunConfig().model
        )
        return {"name": name}

    @app.get("/api/workbench/runs")
    async def runs(request: Request) -> RunCollection:
        """Return recent tasks without losing the active-task identity."""
        runtime = runtime_from(request)
        active = runtime.runner.active
        return RunCollection(
            runs=await asyncio.to_thread(runtime.store.list_runs),
            active_id=runtime.runner.active_id,
            runtime_error=active.failure if active else None,
        )

    @app.post("/api/workbench/runs", status_code=201)
    async def start_run(request: Request, config: RunConfig) -> RunRecord:
        """Start an explicitly requested workload on the server."""
        return await runtime_from(request).runner.start(config)

    @app.delete("/api/workbench/runs/completed")
    async def clear_completed(request: Request) -> ClearResult:
        """Clear all terminal tasks while retaining inputs and service logs."""
        return ClearResult(cleared=await runtime_from(request).runner.cleanup(None))

    @app.get("/api/workbench/runs/{run_id}")
    async def run_detail(request: Request, run_id: str) -> RunDetail:
        """Provide durable summary statistics and recent task events."""
        store = runtime_from(request).store
        run, statistics, timeline, events = await asyncio.gather(
            asyncio.to_thread(store.get_run, run_id),
            asyncio.to_thread(store.statistics, run_id),
            asyncio.to_thread(store.timeline, run_id),
            asyncio.to_thread(store.events, run_id),
        )
        return RunDetail(
            run=run, statistics=statistics, timeline=timeline, events=events
        )

    @app.post("/api/workbench/runs/{run_id}/stop")
    async def stop_run(request: Request, run_id: str) -> RunRecord:
        """Stop dispatch and return the persisted lifecycle state."""
        runtime = runtime_from(request)
        await runtime.runner.stop(run_id)
        return await asyncio.to_thread(runtime.store.get_run, run_id)

    @app.delete("/api/workbench/runs/{run_id}")
    async def clear_run(request: Request, run_id: str) -> ClearResult:
        """Delete one finished test, including its sampled audio."""
        return ClearResult(cleared=await runtime_from(request).runner.cleanup(run_id))

    @app.get("/api/workbench/runs/{run_id}/requests")
    async def request_records(
        request: Request,
        run_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(30, ge=1, le=200),
        failures_only: bool = False,
    ) -> RequestPage:
        """Return one page of all attempts or terminal failures."""
        store = runtime_from(request).store
        await asyncio.to_thread(store.get_run, run_id)
        return await asyncio.to_thread(
            store.request_page, run_id, offset, limit, failures_only
        )

    @app.get("/api/workbench/requests")
    async def search_requests(
        request: Request,
        task_name: str = Query(min_length=1, max_length=100),
        offset: int = Query(0, ge=0),
        limit: int = Query(30, ge=1, le=200),
        failures_only: bool = False,
    ) -> RequestPage:
        """Search all stored task names without restricting the history selector window."""
        return await asyncio.to_thread(
            runtime_from(request).store.request_page,
            None,
            offset,
            limit,
            failures_only,
            task_name,
        )

    @app.get("/api/workbench/requests/{request_id}/http")
    async def request_http(request: Request, request_id: str) -> HttpRequestDetails:
        """Return captured inputs or clearly label a best-effort historical reconstruction."""
        runtime = runtime_from(request)
        record = await asyncio.to_thread(runtime.store.get_request, request_id)
        snapshot = await asyncio.to_thread(runtime.store.http_requests.get, request_id)
        if snapshot is not None:
            return snapshot
        else:
            run = await asyncio.to_thread(runtime.store.get_run, record.run_id)
            line = runtime.dataset.lines[record.line_id]
            try:
                reference_uri = await asyncio.to_thread(
                    runtime.dataset.reference_uri, record.line_id
                )
            except (OSError, ValueError) as exception:
                reference_uri = f"Reference audio unavailable: {exception}"
            return HttpRequestDetails(
                url=f"{settings.api_base.rstrip('/')}/v1/audio/speech",
                body=create_speech_payload(
                    run.config.model,
                    record.text,
                    run.config.language,
                    reference_uri,
                    line.text_zh,
                ),
                source="reconstructed",
                note="此记录没有请求快照；以下按历史配置、当前素材和服务地址还原，不能确认与当时发出的参数完全一致。",
            )

    @app.get("/api/workbench/audio/{request_id}")
    async def generated_audio(request: Request, request_id: str) -> FileResponse:
        """Play only audio linked to a known stored request."""
        store = runtime_from(request).store
        record = await asyncio.to_thread(store.get_request, request_id)
        if not record.audio_path:
            raise HTTPException(404, "This request has no retained audio sample")
        else:
            path = (store.root / record.audio_path).resolve()
            if not path.is_relative_to(store.audio_root) or not path.is_file():
                raise HTTPException(404, "The audio sample is unavailable")
            else:
                return FileResponse(path, media_type="audio/wav")

    @app.get("/api/workbench/references/{line_id}")
    async def reference_audio(request: Request, line_id: str) -> FileResponse:
        """Play original reference audio without permitting arbitrary file reads."""
        source = runtime_from(request).dataset
        line = source.lines[line_id]
        return FileResponse(source.resolve(line.audio_path), media_type="audio/wav")

    @app.get("/api/workbench/gpu")
    async def gpu(request: Request, run_id: str | None = None) -> GpuView:
        """Combine current telemetry with a selected test's persisted history."""
        runtime = runtime_from(request)
        if run_id:
            history = await asyncio.to_thread(runtime.store.gpu_history, run_id)
        else:
            history = await asyncio.to_thread(runtime.store.recent_gpu_history)
        return GpuView(current=runtime.gpu.current, history=history)

    @app.get("/api/workbench/logs")
    async def logs(
        request: Request,
        cursor: str | None = Query(None, max_length=160),
        limit: int = Query(DEFAULT_LOG_LINES, ge=1, le=MAX_LOG_LINES),
        request_id: str | None = Query(None, min_length=1, max_length=200),
        scope: Literal["all", "system", "inference"] = "all",
        keyword: str = Query("", max_length=200),
    ) -> LogTail:
        """Follow recent logs or complete records for an exact service request."""
        return await asyncio.to_thread(
            runtime_from(request).logs.tail, cursor, request_id, limit, scope, keyword
        )

    @app.get("/api/workbench/logs/search")
    async def search_logs(
        request: Request,
        request_id: str = Query("", max_length=200),
        keyword: str = Query("", max_length=200),
        location: str = Query("", max_length=200),
        scope: Literal["all", "system", "inference"] = "all",
        limit: int = Query(DEFAULT_LOG_LINES, ge=1, le=MAX_LOG_LINES),
        snapshot: int | None = Query(None, ge=0),
        before: int | None = Query(None, ge=1),
    ) -> LogSearchPage:
        """查询归档中的完整日志记录，并用快照固定分页结果。"""
        archive = runtime_from(request).logs
        page = await asyncio.to_thread(
            search_log_history,
            archive.database_path,
            request_id.strip(),
            keyword.strip(),
            location.strip(),
            scope,
            limit,
            snapshot,
            before,
        )
        page.warning = archive.error
        return page

    @app.get("/api/workbench/logs/download")
    async def download_logs(
        request: Request,
        request_id: str = Query(min_length=1, max_length=200),
    ) -> StreamingResponse:
        """Export all archived matching records, including logs from before file rotation."""
        filename = re.sub(r"[^A-Za-z0-9_.-]", "_", request_id)
        return StreamingResponse(
            runtime_from(request).logs.download(request_id),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}.log"'},
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")
    return app
