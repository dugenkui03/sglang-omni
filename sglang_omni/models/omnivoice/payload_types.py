# SPDX-License-Identifier: Apache-2.0
"""阶段间状态；Tensor 由现有 relay 传递，不转为 CPU 列表。"""

from dataclasses import dataclass, field

import torch

from sglang_omni.models.omnivoice.hf_config import GenerationOptions, VocoderOptions
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase


@dataclass(kw_only=True)
class OmniVoiceState(DeclarativeStateBase):
    """同一请求的参考信息、生成设置及分段码本。"""

    text: str = ""
    reference_text: str = ""
    language: str | None = None
    speed: float = 1.0
    seed: int | None = None
    reference_waveform: torch.Tensor | None = None
    reference_codes: torch.Tensor | None = None
    reference_rms: float = 0.0
    generation: GenerationOptions = field(default_factory=GenerationOptions)
    vocoder: VocoderOptions = field(default_factory=VocoderOptions)
    chunk_texts: list[str] = field(default_factory=list)
    target_frames: list[int] = field(default_factory=list)
    generated_codes: list[torch.Tensor] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """将参数模型转为 wire 字典，保留 Tensor 的设备。"""
        values = super().to_dict()
        values["generation"] = self.generation.model_dump()
        values["vocoder"] = self.vocoder.model_dump()
        return values

    @classmethod
    def from_dict(cls, data: object) -> "OmniVoiceState":
        """从阶段载荷恢复经过校验的参数模型。"""
        state = super().from_dict(data)
        state.generation = GenerationOptions.model_validate(state.generation)
        state.vocoder = VocoderOptions.model_validate(state.vocoder)
        return state
