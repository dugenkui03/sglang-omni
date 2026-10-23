# SPDX-License-Identifier: Apache-2.0
"""测试真实请求转换及不额外筛掉短参考音频的约束。"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from sglang_omni.models.omnivoice.payload_types import OmniVoiceState
from sglang_omni.models.omnivoice.request_builders import build_omnivoice_state
from sglang_omni.proto import OmniRequest, StagePayload


@pytest.fixture
def reference(tmp_path: Path) -> str:
    """只有 10ms 的有效音频仍应交给模型编码。"""
    path = tmp_path / "reference.wav"
    sf.write(path, np.full(240, 0.2, dtype=np.float32), 24000)
    return str(path)


def payload(reference: str, structured: bool = False) -> StagePayload:
    """构造两种入口以及默认关闭静音处理的最小请求。"""
    inputs = {"text": "Hello"}
    if structured:
        inputs["references"] = [{"audio_path": reference, "text": "你好"}]
    else:
        inputs.update(ref_audio=reference, ref_text="你好")
    return StagePayload(
        request_id="speech-test",
        request=OmniRequest(
            inputs=inputs,
            params={"stage_params": {"preprocessing": {"preprocess_prompt": False}}},
        ),
        data={},
    )


@pytest.mark.parametrize("structured", [False, True])
def test_reference_formats_and_short_audio(reference: str, structured: bool) -> None:
    """保留原始短音频，默认参数使用官方值。"""
    state = build_omnivoice_state(payload(reference, structured), 24000)
    assert state.reference_waveform.shape == (1, 240)
    assert state.generation.num_step == 32
    assert state.reference_text == "你好"
    assert state.generation.guidance_scale == 2.0


def test_overrides_and_tensor_state_roundtrip(reference: str) -> None:
    """显式的零温度不可被默认值覆盖，状态往返不拷贝 Tensor。"""
    request = payload(reference)
    request.request.params["stage_params"].update(
        tts_engine={"num_step": 3, "position_temperature": 0.0},
        vocoder={"postprocess_output": False},
    )
    request.request.metadata["tts_params"] = {
        "language": "English",
        "seed": 9,
        "speed": 2.0,
    }
    state = build_omnivoice_state(request, 24000)
    state.reference_codes = torch.ones(8, 5, dtype=torch.long)
    restored = OmniVoiceState.from_dict(state.to_dict())
    assert restored.reference_codes is state.reference_codes
    assert restored.reference_waveform is state.reference_waveform
    assert restored.generation.num_step == 3
    assert restored.generation.position_temperature == 0
    assert not restored.vocoder.postprocess_output
    assert restored.language == "en" and restored.seed == 9


@pytest.mark.parametrize(
    "settings",
    [
        {"num_step": 0},
        {"num_step": True},
        {"guidance_scale": float("nan")},
        {"duration": -1},
        {"typo": 3},
    ],
)
def test_invalid_model_options_fail_before_audio_loading(
    reference: str, settings: dict, monkeypatch
) -> None:
    """参数拼错和无效数值应在进入 GPU 前被拒绝。"""
    request = payload(reference)
    request.request.params["stage_params"]["tts_engine"] = settings

    def unexpected(*args, **kwargs):
        """无效参数不应该触发媒体读取。"""
        raise AssertionError("audio was loaded")

    monkeypatch.setattr(
        "sglang_omni.models.omnivoice.request_builders.load_audio", unexpected
    )
    with pytest.raises(ValueError):
        build_omnivoice_state(request, 24000)


def test_missing_transcript_and_multiple_references(reference: str) -> None:
    """缺少参考文本不自动调用 ASR。"""
    request = payload(reference)
    request.request.inputs.pop("ref_text")
    with pytest.raises(ValueError, match="transcript"):
        build_omnivoice_state(request, 24000)
    request = payload(reference, True)
    request.request.inputs["references"] *= 2
    with pytest.raises(ValueError, match="exactly one"):
        build_omnivoice_state(request, 24000)


@pytest.mark.parametrize("structured", [False, True])
def test_speech_service_lowering(reference: str, structured: bool) -> None:
    """经过真实 HTTP 适配层后，参考元数据不应被误判为重复音频。"""
    from sglang_omni.client.client import Client
    from sglang_omni.serve.protocol import CreateSpeechRequest
    from sglang_omni.serve.speech_service import SpeechRequestValidator

    fields = (
        {"references": [{"audio_path": reference, "text": "你好"}]}
        if structured
        else {"ref_audio": reference, "ref_text": "你好"}
    )
    request = CreateSpeechRequest(
        input="Hello",
        language="English",
        seed=7,
        stage_params={"preprocessing": {"preprocess_prompt": False}},
        **fields,
    )
    service = SpeechRequestValidator(
        default_model="k2-fsa/OmniVoice",
        allowed_local_media_path=Path(reference).parent,
        required_speech_reference_count=1,
        speech_reference_text_required=True,
    )
    generated = service.build_generate_request(request)
    state = build_omnivoice_state(
        StagePayload(
            request_id="speech-lowered",
            request=Client.build_omni_request(generated),
            data={},
        ),
        24000,
    )
    assert state.language == "en" and state.seed == 7
    assert state.reference_text == "你好"
