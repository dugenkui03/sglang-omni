# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xiaomi Corp. (authors: Han Zhu). See LICENSE.
"""检查点配置与请求级生成参数。"""

from pydantic import BaseModel, ConfigDict, Field
from transformers import PretrainedConfig, Qwen3Config


class OmniVoiceConfig(PretrainedConfig):
    """保留官方检查点的嵌套 Qwen3 配置及码本字段。"""

    model_type = "omnivoice"
    sub_configs = {"llm_config": Qwen3Config}

    def __init__(
        self,
        audio_vocab_size: int = 1025,
        audio_mask_id: int = 1024,
        num_audio_codebook: int = 8,
        audio_codebook_weights: list[float] | None = None,
        llm_config: dict[str, object] | Qwen3Config | None = None,
        **kwargs: object,
    ) -> None:
        """将检查点 JSON 转为具体配置，保持权重加载协议。"""
        if llm_config is None:
            llm_config = Qwen3Config()
        elif isinstance(llm_config, dict):
            llm_config = Qwen3Config(**llm_config)
        self.llm_config = llm_config
        super().__init__(**kwargs)
        self.audio_vocab_size = audio_vocab_size
        self.audio_mask_id = audio_mask_id
        self.num_audio_codebook = num_audio_codebook
        self.audio_codebook_weights = audio_codebook_weights or [8, 8, 6, 6, 4, 4, 2, 2]


class GenerationOptions(BaseModel):
    """请求边界校验后不再改变的官方采样设置。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    num_step: int = Field(default=32, gt=0, strict=True)
    guidance_scale: float = Field(default=2.0, ge=0, strict=True)
    t_shift: float = Field(default=0.1, gt=0, strict=True)
    layer_penalty_factor: float = Field(default=5.0, ge=0, strict=True)
    position_temperature: float = Field(default=5.0, ge=0, strict=True)
    class_temperature: float = Field(default=0.0, ge=0, strict=True)
    denoise: bool = Field(default=True, strict=True)
    duration: float | None = Field(default=None, gt=0, strict=True)
    audio_chunk_duration: float = Field(default=15.0, gt=0, strict=True)
    audio_chunk_threshold: float = Field(default=30.0, gt=0, strict=True)


class PreprocessingOptions(BaseModel):
    """控制官方参考音频处理，不增加时长筛选。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    preprocess_prompt: bool = Field(default=True, strict=True)


class VocoderOptions(BaseModel):
    """控制完整波形的静音处理、淡入淡出和边缘填充。"""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    postprocess_output: bool = Field(default=True, strict=True)
    pad_duration: float = Field(default=0.1, ge=0, strict=True)
    fade_duration: float = Field(default=0.1, ge=0, strict=True)
