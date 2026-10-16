"""Receive speech responses without imposing audio-quality acceptance rules."""

import io
import time
from dataclasses import dataclass

import httpx
import soundfile

MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_ERROR_BYTES = 8192


@dataclass
class ResponseData:
    """Keep only headers required for request accounting."""

    status_code: int
    body: bytes
    request_id: str | None
    finish_reason: str | None
    latency_ms: float


def create_speech_payload(
    model: str, text: str, language: str, reference_uri: str, reference_text: str
) -> dict[str, object]:
    """Build the JSON body from original text and reference bytes without filtering."""
    return {
        "model": model,
        "input": text,
        "language": language,
        "response_format": "wav",
        "stream": False,
        "references": [{"audio_path": reference_uri, "text": reference_text}],
    }


def read_audio_duration(audio_bytes: bytes) -> float | None:
    """Read optional metadata without changing the server-reported outcome."""
    try:
        with soundfile.SoundFile(io.BytesIO(audio_bytes)) as audio:
            return audio.frames / audio.samplerate
    except (ValueError, soundfile.LibsndfileError):
        return None


async def receive_audio(
    client: httpx.AsyncClient, api_base: str, payload: dict[str, object]
) -> ResponseData:
    """Read complete audio with a size cap and a bounded error-body excerpt."""
    started_seconds = time.monotonic()
    async with client.stream(
        "POST",
        f"{api_base.rstrip('/')}/v1/audio/speech",
        json=payload,
    ) as response:
        body = bytearray()
        maximum = MAX_RESPONSE_BYTES if response.is_success else MAX_ERROR_BYTES
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > maximum:
                if response.is_success:
                    raise ValueError("Response exceeded the 32 MiB audio limit")
                else:
                    body.extend(chunk[: maximum - len(body)])
                    break
            else:
                body.extend(chunk)
        return ResponseData(
            status_code=response.status_code,
            body=bytes(body),
            request_id=response.headers.get("X-Request-Id"),
            finish_reason=response.headers.get("X-Finish-Reason", "unknown"),
            latency_ms=(time.monotonic() - started_seconds) * 1000,
        )
