"""Provide synthetic, isolated fixtures only for workbench contract tests."""

import hashlib
import io
import json
import math
import struct
import wave
from pathlib import Path

import pytest

from tts_workbench.config import Settings


@pytest.fixture
def wav_bytes() -> bytes:
    """Generate a small tone to validate audio transport without TTS inference."""
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(
            b"".join(
                struct.pack("<h", int(2000 * math.sin(index / 8)))
                for index in range(4800)
            )
        )
    return output.getvalue()


@pytest.fixture
def settings(tmp_path: Path, wav_bytes: bytes) -> Settings:
    """Build a two-line portable dataset outside the result directory."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "reference.wav").write_bytes(wav_bytes)
    rows = [
        {
            "line_id": f"unit_{index}",
            "unit_id": "unit",
            "line_index": index,
            "speaker": "角色",
            "speaker_id": "unit:role",
            "text_zh": f"中文台词{index}",
            "text_en_tts": f"English line {index}",
            "audio_path": "reference.wav",
            "audio_sha256": hashlib.sha256(wav_bytes).hexdigest(),
            "reference_duration_ms": 200,
            "reference_text_needs_review": index == 1,
        }
        for index in range(2)
    ]
    (root / "lines.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "synthetic-test-only",
                "line_count": 2,
                "units": [
                    {
                        "unit_id": "unit",
                        "title": "测试单元",
                        "lines_path": "lines.jsonl",
                        "line_count": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    log_file = tmp_path / "tts.log"
    log_file.write_text("original service log\n", encoding="utf-8")
    return Settings(
        api_base="http://tts.test",
        data_directory=root,
        results_directory=tmp_path / "results",
        log_file=log_file,
    )
