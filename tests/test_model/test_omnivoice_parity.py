# SPDX-License-Identifier: Apache-2.0
"""显式启用的官方检查点对齐；默认不下载模型或使用 GPU。"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import AutoTokenizer

from sglang_omni.models.omnivoice.audio import load_codec
from sglang_omni.models.omnivoice.generation import SamplingItem, prepare_inputs
from sglang_omni.models.omnivoice.model import OmniVoiceModel
from sglang_omni.models.omnivoice.payload_types import OmniVoiceState
from sglang_omni.models.omnivoice.request_builders import build_omnivoice_state
from sglang_omni.models.omnivoice.stages import (
    decode_batch,
    encode_batch,
    generate_batch,
)
from sglang_omni.proto import OmniRequest, StagePayload

pytestmark = pytest.mark.accelerator
SOURCE_REVISION = "08be0b4ccbac3e13e374e86fbfead4b4cac343e2"


@pytest.fixture(scope="module")
def models():
    """只读取指定版本的参考仓库，测试不修改其代码或虚拟环境。"""
    source = os.environ.get("OMNIVOICE_UPSTREAM_SOURCE")
    checkpoint = os.environ.get("OMNIVOICE_PARITY_CHECKPOINT")
    reference = os.environ.get("OMNIVOICE_PARITY_REFERENCE")
    reference_text = os.environ.get("OMNIVOICE_PARITY_REFERENCE_TEXT")
    if not all((source, checkpoint, reference, reference_text)):
        pytest.skip(
            "Set OMNIVOICE_UPSTREAM_SOURCE, OMNIVOICE_PARITY_CHECKPOINT, OMNIVOICE_PARITY_REFERENCE and OMNIVOICE_PARITY_REFERENCE_TEXT"
        )
    revision = subprocess.check_output(
        ["git", "-C", source, "rev-parse", "HEAD"], text=True
    ).strip()
    assert (
        revision == SOURCE_REVISION
    ), f"Expected reference source {SOURCE_REVISION}, got {revision}"
    device = os.environ.get("OMNIVOICE_PARITY_DEVICE", "cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip(
            "CUDA unavailable; set OMNIVOICE_PARITY_DEVICE=cpu for numerical checks"
        )
    sys.path.insert(0, source)
    try:
        official_class = importlib.import_module("omnivoice").OmniVoice
        upstream = official_class.from_pretrained(
            checkpoint,
            device_map=device,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        native = (
            OmniVoiceModel.from_pretrained(
                checkpoint, dtype=torch.float32, attn_implementation="sdpa"
            )
            .to(device)
            .eval()
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        codec = load_codec(checkpoint, device)
        yield upstream, native, tokenizer, codec, reference, reference_text
    finally:
        sys.path.remove(source)
        load_codec.cache_clear()


@pytest.mark.parametrize(
    "text,language,chunked,sampled",
    [
        pytest.param("Your Majesty.", "en", False, False, id="english"),
        pytest.param("你好，欢迎回来。", "zh", False, False, id="chinese"),
        pytest.param(
            "Hello there. Welcome home. Goodbye now.", "en", True, False, id="chunked"
        ),
        pytest.param("Your Majesty.", "en", False, True, id="default-sampling"),
    ],
)
@torch.inference_mode()
def test_official_stage_and_waveform_parity(
    models,
    monkeypatch,
    tmp_path: Path,
    text: str,
    language: str,
    chunked: bool,
    sampled: bool,
) -> None:
    """比较官方输入、参考码本、目标长度、首轮 logits、生成码本和最终波形。"""
    upstream, native, tokenizer, codec, reference, reference_text = models
    options = {"num_step": 4, "position_temperature": 0.0, "class_temperature": 0.0}
    if sampled:
        options = {}
    if chunked:
        options.update(audio_chunk_threshold=0.1, audio_chunk_duration=0.2)
    expected_codes: list[torch.Tensor] = []
    expected_logits: list[torch.Tensor] = []
    actual_logits: list[torch.Tensor] = []
    original_iterative = (
        upstream._generate_iterative
    )  # noqa: leading-underscore  # 官方对照接口

    def record_iterative(task, config):
        """记录官方每段结果，不改变计算。"""
        result = original_iterative(task, config)
        expected_codes.extend(code.detach().cpu() for code in result)
        return result

    def capture_expected(module, arguments, output):
        """仅保存首轮，避免测试自身占用大量 CPU 内存。"""
        if not expected_logits:
            expected_logits.append(output.logits.detach().cpu())

    def capture_actual(module, arguments, output):
        """保存原生实现的首轮 logits。"""
        if not actual_logits:
            actual_logits.append(output.detach().cpu())

    monkeypatch.setattr(upstream, "_generate_iterative", record_iterative)
    expected_hook = upstream.register_forward_hook(capture_expected)
    actual_hook = native.register_forward_hook(capture_actual)
    try:
        prompt = upstream.create_voice_clone_prompt(reference, reference_text)
        payload = StagePayload(
            request_id=f"parity-{language}",
            request=OmniRequest(
                inputs={
                    "text": text,
                    "ref_audio": reference,
                    "ref_text": reference_text,
                },
                params={
                    "seed": 1234,
                    "language": language,
                    "stage_params": {"tts_engine": options},
                },
            ),
            data={},
        )
        state = build_omnivoice_state(payload, codec.model.config.sample_rate)
        payload.data = state.to_dict()
        encoded = encode_batch(
            [payload],
            lambda _: False,
            codec=codec,
            codebooks=native.config.num_audio_codebook,
        )[0]
        assert isinstance(encoded, StagePayload)
        state = OmniVoiceState.from_dict(encoded.data)
        torch.testing.assert_close(
            state.reference_codes.cpu(), prompt.ref_audio_tokens.cpu(), rtol=0, atol=0
        )
        assert state.reference_text == prompt.ref_text
        item = SamplingItem(
            request_id=payload.request_id,
            state=state,
            text=state.chunk_texts[0],
            target_frames=state.target_frames[0],
            generator=torch.Generator(device=native.device).manual_seed(1234),
        )
        input_ids, audio_mask = prepare_inputs(item, native, tokenizer)
        expected_inputs = upstream._prepare_inference_inputs(  # noqa: leading-underscore  # 官方对照接口
            item.text,
            item.target_frames,
            ref_text=prompt.ref_text,
            ref_audio_tokens=prompt.ref_audio_tokens,
            lang=language,
        )
        torch.testing.assert_close(
            input_ids, expected_inputs["input_ids"][0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            audio_mask, expected_inputs["audio_mask"][0], rtol=0, atol=0
        )
        devices = [native.device] if native.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(1234)
            expected = upstream.generate(
                text=text, language=language, voice_clone_prompt=prompt, **options
            )[0]
        generated = generate_batch(
            [encoded], lambda _: False, model=native, tokenizer=tokenizer
        )[0]
        assert isinstance(generated, StagePayload)
        actual_codes = OmniVoiceState.from_dict(generated.data).generated_codes
        assert [codes.shape[-1] for codes in expected_codes] == state.target_frames
        assert len(actual_codes) == len(expected_codes)
        torch.testing.assert_close(
            actual_logits[0], expected_logits[0], rtol=1e-4, atol=1e-5
        )
        for actual, expected_code in zip(actual_codes, expected_codes, strict=True):
            torch.testing.assert_close(actual.cpu(), expected_code, rtol=0, atol=0)
        decoded = decode_batch([generated], lambda _: False, codec=codec)[0]
        assert isinstance(decoded, StagePayload)
        actual = np.frombuffer(decoded.data["audio_waveform"], dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
        output = Path(os.environ.get("OMNIVOICE_PARITY_OUTPUT_DIR", str(tmp_path)))
        output.mkdir(parents=True, exist_ok=True)
        name = (
            f"{language}-{'chunked' if chunked else 'sampled' if sampled else 'short'}"
        )
        sf.write(output / f"{name}.wav", actual, state.sample_rate)
        (output / f"{name}.json").write_text(
            json.dumps(
                {
                    "source_revision": SOURCE_REVISION,
                    "device": str(native.device),
                    "torch": torch.__version__,
                    "frames": state.target_frames,
                    "sample_rate": state.sample_rate,
                    "max_waveform_error": float(np.max(np.abs(actual - expected))),
                    "options": options,
                },
                indent=2,
            )
        )
    finally:
        expected_hook.remove()
        actual_hook.remove()
