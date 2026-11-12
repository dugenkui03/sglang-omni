# SPDX-License-Identifier: Apache-2.0
"""将 speech 请求转换为 OmniVoice 的类型化阶段状态。"""

import json
import logging
import secrets
from dataclasses import fields

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, JsonValue

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


logger = logging.getLogger(__name__)


def log_fields(value: object, *, audio_reference: bool = False) -> JsonValue:
    """保留请求字段，音频编码内容和数组仅输出长度或形状，避免日志膨胀。"""
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"type": type(value).__name__, "bytes": memoryview(value).nbytes}
    if isinstance(value, str):
        if value.startswith("data:"):
            return {"type": "data_uri", "characters": len(value)}
        return value
    if isinstance(value, dict):
        return {
            str(name): (
                {"type": "base64", "characters": len(item)}
                if audio_reference and name == "data" and isinstance(item, str)
                else log_fields(item, audio_reference=name == "references")
            )
            for name, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [log_fields(item, audio_reference=audio_reference) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, BaseModel):
        return log_fields(value.model_dump())
    return {"type": type(value).__name__}


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
    logger.info(
        f"request_id={payload.request_id} OmniVoice 预处理输入（省略音频内容）\n"
        f"{json.dumps(log_fields({'request': payload.request.to_dict(), 'data': payload.data, 'sample_rate': sample_rate}), ensure_ascii=False, indent=2)}"
    )
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
        or audio_data_uri_from_reference(reference) # tip 原始请求从这里获取音频对应的二进制
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
    # NOTE 获取 waveform
    waveform = load_audio(
        source,
        source_name="OmniVoice reference", target_sample_rate=sample_rate
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
    state = OmniVoiceState(
        sample_rate=sample_rate, # 模型配置
        text=text, # tts 文本
        reference_text=reference_text, # 参考音频对应的文本
        language=language, # 目标语言
        speed=controls.speed, # 速度
        seed=controls.seed, # 推理种子
        reference_waveform=torch.from_numpy(np.ascontiguousarray(waveform)), # tip 音频
        reference_rms=reference_rms, # 音量均方根 np.sqrt(np.mean(waveform ** 2))，waveform 是个元素 list、类似于 [0.1, -0.2,......, 0.x]
        #tip 音频生成配置
        # {
        #  "num_step": 32,
        #  "denoise": true,
        #  "duration": 1.2
        #  ......
        # }
        generation=generation,
        # NOTE vocoder 叫 声码器，负责音频码本解码为波形(Waveform)
        #   音频码本格式是 [8, seq_len], seq_len 表示音频帧数
        vocoder=vocoder,
    )
    logger.info(
        f"request_id={payload.request_id} OmniVoice 预处理状态（省略音频内容）\n"
        f"{json.dumps(log_fields({field.name: getattr(state, field.name) for field in fields(state)}), ensure_ascii=False, indent=2)}"
    )
    return state
