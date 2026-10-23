# SPDX-License-Identifier: Apache-2.0
"""将 speech 请求转换为 OmniVoice 的类型化阶段状态。"""

import secrets

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from sglang_omni.models.omnivoice.audio import remove_silence
from sglang_omni.models.omnivoice.hf_config import (
    GenerationOptions,
    PreprocessingOptions,
    VocoderOptions,
)
from sglang_omni.models.omnivoice.payload_types import OmniVoiceState
from sglang_omni.models.omnivoice.text import add_punctuation, resolve_language
from sglang_omni.proto import StagePayload
from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.audio_payload import audio_data_uri_from_reference


class RequestControls(BaseModel):
    """校验非模型采样字段，拒绝布尔值伪装成数字。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    speed: float = Field(default=1.0, gt=0, strict=True)
    seed: int = Field(
        default_factory=lambda: secrets.randbits(63), ge=0, lt=2**64, strict=True
    )
    language: str | None = Field(default=None, strict=True)


def mapping(value: object, name: str) -> dict[str, object]:
    """在请求边界把外部对象收窄成字符串键字典。"""
    if value is None:
        return {}
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"OmniVoice {name} must be an object")
    return value


def build_omnivoice_state(payload: StagePayload, sample_rate: int) -> OmniVoiceState:
    """先校验输入及参数，再读取音频；模型不自动补参考文本。"""
    params = payload.request.params
    inputs = payload.request.inputs
    inputs = {"text": inputs} if isinstance(inputs, str) else mapping(inputs, "inputs")
    tts = mapping(payload.request.metadata.get("tts_params"), "tts_params")
    if params.get("stream"):
        raise ValueError("OmniVoice supports stream=false only")
    if tts.get("instructions") or tts.get("task_type") not in (None, "Base"):
        raise ValueError("OmniVoice currently supports reference voice cloning only")
    for name in (
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "max_new_tokens",
    ):
        if params.get(name) is not None:
            raise ValueError(
                f"OmniVoice does not use {name}; use stage_params.tts_engine"
            )
    text = inputs.get("text", inputs.get("input"))
    if not isinstance(text, str) or not text.strip():
        raise ValueError("OmniVoice requires nonempty input text")
    references = inputs.get("references") or tts.get("references") or []
    if not isinstance(references, list) or len(references) > 1:
        raise ValueError("OmniVoice requires exactly one reference")
    flat_audio = inputs.get("ref_audio") or tts.get("ref_audio")
    if references and inputs.get("ref_audio"):
        raise ValueError("OmniVoice accepts either references or ref_audio, not both")
    reference = mapping(references[0], "reference") if references else {}
    source = (
        reference.get("audio_path")
        or reference.get("ref_audio")
        or reference.get("audio")
        or audio_data_uri_from_reference(reference)
        or flat_audio
    )
    reference_text = (
        reference.get("text")
        if references
        else inputs.get("ref_text", tts.get("ref_text"))
    )
    if not isinstance(source, str) or not source:
        raise ValueError("OmniVoice requires reference audio")
    if not isinstance(reference_text, str) or not reference_text.strip():
        raise ValueError("OmniVoice requires a reference transcript")
    stages = mapping(params.get("stage_params"), "stage_params")
    unknown = stages.keys() - {"preprocessing", "tts_engine", "vocoder"}
    if unknown:
        raise ValueError(f"Unknown OmniVoice stage parameters: {sorted(unknown)}")
    generation = GenerationOptions.model_validate(
        mapping(stages.get("tts_engine"), "tts_engine")
    )
    preprocessing = PreprocessingOptions.model_validate(
        mapping(stages.get("preprocessing"), "preprocessing")
    )
    vocoder = VocoderOptions.model_validate(mapping(stages.get("vocoder"), "vocoder"))
    controls = RequestControls.model_validate(
        {
            key: value
            for key, value in {
                "speed": tts.get("speed", params.get("speed", 1.0)),
                "seed": tts.get("seed", params.get("seed")),
                "language": tts.get("language", params.get("language")),
            }.items()
            if value is not None
        }
    )
    language = resolve_language(controls.language)
    waveform = load_audio(
        source, source_name="OmniVoice reference", target_sample_rate=sample_rate
    ).reshape(1, -1)
    reference_rms = float(np.sqrt(np.mean(waveform**2)))
    if 0 < reference_rms < 0.1:
        waveform = waveform * 0.1 / reference_rms
    if preprocessing.preprocess_prompt:
        waveform = remove_silence(
            waveform, sample_rate, mid_sil=200, lead_sil=100, trail_sil=200
        )
        reference_text = add_punctuation(reference_text)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(
            "OmniVoice reference audio is empty or non-finite after preprocessing"
        )
    return OmniVoiceState(
        sample_rate=sample_rate,
        text=text,
        reference_text=reference_text,
        language=language,
        speed=controls.speed,
        seed=controls.seed,
        reference_waveform=torch.from_numpy(np.ascontiguousarray(waveform)),
        reference_rms=reference_rms,
        generation=generation,
        vocoder=vocoder,
    )
