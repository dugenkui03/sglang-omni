"""Benchmark voice cloning on the GPU worker with fixed HTTP concurrency."""

import argparse
import asyncio
import csv
import io
import json
import statistics
import subprocess
import time
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypedDict

import httpx


class SpeechRequest(TypedDict):
    model: str
    input: str
    voice: str
    language: Literal["Chinese", "English"]
    task_type: str
    ref_audio: str
    ref_text: str
    x_vector_only_mode: bool
    stream: bool
    stream_codec_output: bool
    response_format: str
    max_new_tokens: int
    seed: int


@dataclass(kw_only=True)
class RequestCounters:
    active_requests: int = 0
    peak_active_requests: int = 0


@dataclass(kw_only=True)
class RequestResult:
    request_number: int
    language: str
    started_at_utc: str
    latency_seconds: float
    status_code: int | None
    request_id: str | None
    audio_duration_seconds: float | None
    real_time_factor: float | None
    audio_file: str | None
    error: str | None


def percentile(values: list[float], percentage: float) -> float:
    ordered_values = sorted(values)
    position = (len(ordered_values) - 1) * percentage / 100
    lower_position = int(position)
    upper_position = min(lower_position + 1, len(ordered_values) - 1)
    return ordered_values[lower_position] + (
        ordered_values[upper_position] - ordered_values[lower_position]
    ) * (position - lower_position)


async def send_request(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    counters: RequestCounters,
    request_number: int,
    payload: SpeechRequest,
    audio_directory: Path | None,
) -> RequestResult:
    async with semaphore:
        counters.active_requests += 1
        counters.peak_active_requests = max(
            counters.peak_active_requests, counters.active_requests
        )
        started_at_utc = datetime.now(timezone.utc).isoformat()
        started_seconds = time.perf_counter()
        status_code = None
        request_id = None
        try:
            response = await client.post("/v1/audio/speech", json=payload)
            latency_seconds = time.perf_counter() - started_seconds
            status_code = response.status_code
            request_id = response.headers.get("X-SGLang-Omni-Speech-Id")
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}: {response.text[:1000]}")
            else:
                pass
            with wave.open(io.BytesIO(response.content), "rb") as audio:
                audio_duration_seconds = audio.getnframes() / audio.getframerate()
                if audio_duration_seconds <= 0 or not audio.readframes(1):
                    raise ValueError("The response contains no audio frames")
                else:
                    pass
            if audio_directory is not None:
                audio_path = audio_directory / f"{request_number:04d}.wav"
                audio_path.write_bytes(response.content)
                audio_file = str(audio_path)
            else:
                audio_file = None
            return RequestResult(
                request_number=request_number,
                language=payload["language"],
                started_at_utc=started_at_utc,
                latency_seconds=latency_seconds,
                status_code=status_code,
                request_id=request_id,
                audio_duration_seconds=audio_duration_seconds,
                real_time_factor=latency_seconds / audio_duration_seconds,
                audio_file=audio_file,
                error=None,
            )
        except (httpx.HTTPError, ValueError, wave.Error, EOFError) as error:
            return RequestResult(
                request_number=request_number,
                language=payload["language"],
                started_at_utc=started_at_utc,
                latency_seconds=time.perf_counter() - started_seconds,
                status_code=status_code,
                request_id=request_id,
                audio_duration_seconds=None,
                real_time_factor=None,
                audio_file=None,
                error=str(error),
            )
        finally:
            counters.active_requests -= 1


async def run_benchmark(arguments: argparse.Namespace) -> None:
    case_records: list[dict[str, str]] = [
        json.loads(line)
        for line in arguments.cases.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not case_records:
        raise ValueError("The cases file is empty")
    else:
        pass
    payloads: list[SpeechRequest] = []
    for request_number in range(arguments.requests):
        case_record = case_records[(request_number // 2) % len(case_records)]
        language_code = "zh" if request_number % 2 == 0 else "en"
        if not Path(case_record["reference_audio"]).is_file():
            raise FileNotFoundError(case_record["reference_audio"])
        else:
            pass
        payloads.append(
            SpeechRequest(
                model=arguments.model,
                input=case_record[f"text_{language_code}"],
                voice="default",
                language="Chinese" if language_code == "zh" else "English",
                task_type="Base",
                ref_audio=case_record["reference_audio"],
                ref_text=case_record["reference_text"],
                x_vector_only_mode=False,
                stream=False,
                stream_codec_output=False,
                response_format="wav",
                max_new_tokens=arguments.max_new_tokens,
                seed=1234,
            )
        )

    output_directory: Path = arguments.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    audio_directory = output_directory / "audio"
    audio_directory.mkdir()
    (output_directory / "requests.jsonl").write_text(
        "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in payloads),
        encoding="utf-8",
    )
    semaphore = asyncio.Semaphore(arguments.concurrency)
    limits = httpx.Limits(
        max_connections=arguments.concurrency,
        max_keepalive_connections=arguments.concurrency,
    )
    async with httpx.AsyncClient(
        base_url=arguments.base_url.rstrip("/"),
        timeout=httpx.Timeout(arguments.timeout_seconds, connect=10),
        limits=limits,
        trust_env=False,
    ) as client:
        health_response = await client.get("/health")
        health_response.raise_for_status()
        reference_payloads = list(
            {
                (payload["ref_audio"], payload["language"]): payload
                for payload in payloads
            }.values()
        )
        warmup_counters = RequestCounters()
        warmup_results = await asyncio.gather(
            *(
                send_request(
                    client,
                    semaphore,
                    warmup_counters,
                    request_number,
                    reference_payloads[request_number % len(reference_payloads)],
                    None,
                )
                for request_number in range(arguments.warmup_requests)
            )
        )
        (output_directory / "warmup_results.jsonl").write_text(
            "".join(
                json.dumps(asdict(result), ensure_ascii=False) + "\n"
                for result in warmup_results
            ),
            encoding="utf-8",
        )
        if any(result.error for result in warmup_results):
            raise RuntimeError(
                f"Warmup failed; inspect {output_directory / 'warmup_results.jsonl'}"
            )
        else:
            print(f"Warmup complete: {len(warmup_results)} requests", flush=True)

        gpu_csv_path = output_directory / "gpu_samples.csv"
        with gpu_csv_path.open("w", encoding="utf-8", newline="") as gpu_output:
            csv.writer(gpu_output).writerow(
                [
                    "timestamp",
                    "gpu_index",
                    "gpu_name",
                    "gpu_utilization_percent",
                    "memory_controller_utilization_percent",
                    "memory_used_mib",
                    "memory_total_mib",
                    "power_draw_w",
                    "temperature_c",
                ]
            )
            gpu_output.flush()
            with (output_directory / "gpu_sampler.stderr.log").open("w") as gpu_errors:
                gpu_sampler = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--id",
                        "0",
                        "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits",
                        f"--loop-ms={arguments.gpu_sample_interval_ms}",
                    ],
                    stdout=gpu_output,
                    stderr=gpu_errors,
                )
                counters = RequestCounters()
                started_at_utc = datetime.now(timezone.utc).isoformat()
                started_seconds = time.perf_counter()
                try:
                    results = await asyncio.gather(
                        *(
                            send_request(
                                client,
                                semaphore,
                                counters,
                                request_number + 1,
                                payload,
                                audio_directory,
                            )
                            for request_number, payload in enumerate(payloads)
                        )
                    )
                    elapsed_seconds = time.perf_counter() - started_seconds
                finally:
                    gpu_sampler.terminate()
                    try:
                        gpu_sampler.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        gpu_sampler.kill()
                        gpu_sampler.wait()

    (output_directory / "results.jsonl").write_text(
        "".join(
            json.dumps(asdict(result), ensure_ascii=False) + "\n" for result in results
        ),
        encoding="utf-8",
    )
    successful_results = [result for result in results if result.error is None]
    latency_by_language: dict[str, dict[str, float] | None] = {}
    for language in ("all", "Chinese", "English"):
        latency_seconds = [
            result.latency_seconds
            for result in successful_results
            if language == "all" or result.language == language
        ]
        if latency_seconds:
            latency_by_language[language] = {
                "mean": statistics.fmean(latency_seconds),
                "p50": percentile(latency_seconds, 50),
                "p95": percentile(latency_seconds, 95),
                "p99": percentile(latency_seconds, 99),
                "max": max(latency_seconds),
            }
        else:
            latency_by_language[language] = None
    with gpu_csv_path.open(encoding="utf-8", newline="") as gpu_input:
        gpu_samples = list(csv.DictReader(gpu_input, skipinitialspace=True))
    if not gpu_samples:
        raise RuntimeError(
            f"No GPU samples recorded; inspect {output_directory / 'gpu_sampler.stderr.log'}"
        )
    else:
        pass
    utilization_percent = [
        float(sample["gpu_utilization_percent"]) for sample in gpu_samples
    ]
    memory_used_mib = [float(sample["memory_used_mib"]) for sample in gpu_samples]
    audio_seconds = sum(result.audio_duration_seconds for result in successful_results)
    summary = {
        "model": arguments.model,
        "gpu": gpu_samples[0]["gpu_name"],
        "started_at_utc": started_at_utc,
        "request_mode": "non-streaming WAV; complete response latency; reference cloning",
        "warmup_requests_excluded": arguments.warmup_requests,
        "unique_reference_audios": len({payload["ref_audio"] for payload in payloads}),
        "target_concurrency": arguments.concurrency,
        "peak_client_requests": counters.peak_active_requests,
        "requests": len(results),
        "successful_requests": len(successful_results),
        "failed_requests": len(results) - len(successful_results),
        "elapsed_seconds": elapsed_seconds,
        "requests_per_second": len(successful_results) / elapsed_seconds,
        "generated_audio_seconds": audio_seconds,
        "audio_seconds_per_wall_second": audio_seconds / elapsed_seconds,
        "latency_seconds": latency_by_language,
        "gpu_sample_interval_ms": arguments.gpu_sample_interval_ms,
        "gpu_samples": len(gpu_samples),
        "gpu_utilization_percent": {
            "mean": statistics.fmean(utilization_percent),
            "p50": percentile(utilization_percent, 50),
            "p95": percentile(utilization_percent, 95),
            "max": max(utilization_percent),
        },
        "gpu_memory_used_mib": {
            "mean": statistics.fmean(memory_used_mib),
            "max": max(memory_used_mib),
            "total": float(gpu_samples[0]["memory_total_mib"]),
        },
    }
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
    (output_directory / "summary.json").write_text(
        summary_json + "\n", encoding="utf-8"
    )
    print(summary_json, flush=True)
    if len(successful_results) != len(results):
        raise SystemExit(1)
    else:
        pass


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument(
        "--cases",
        type=Path,
        default=repository_root
        / "test_data/outputs/qwen3_tts_base_20_zh_en_20261014/cases.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--warmup-requests", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--gpu-sample-interval-ms", type=int, default=1000)
    arguments = parser.parse_args()
    if (
        min(
            arguments.concurrency,
            arguments.requests,
            arguments.max_new_tokens,
            arguments.gpu_sample_interval_ms,
        )
        < 1
        or arguments.warmup_requests < 0
        or arguments.timeout_seconds <= 0
    ):
        parser.error(
            "Counts, timeout and sampling interval must be positive; warmup may be zero"
        )
    else:
        asyncio.run(run_benchmark(arguments))


if __name__ == "__main__":
    main()
else:
    pass
