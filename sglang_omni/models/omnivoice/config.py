# SPDX-License-Identifier: Apache-2.0
"""OmniVoice 的四阶段配置及模型发现入口。"""

from typing import ClassVar

from pydantic import model_validator

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.models.omnivoice.lang_map import LANG_IDS, LANG_NAME_TO_ID

PACKAGE = "sglang_omni.models.omnivoice"


class OmniVoicePipelineConfig(PipelineConfig):
    """共享同一进程中的 Codec，单 GPU 完成参考编码、生成和解码。"""

    architecture: ClassVar[str] = "OmniVoice"
    requires_model_capabilities: ClassVar[bool] = True
    required_speech_reference_count: ClassVar[int] = 1
    speech_reference_text_required: ClassVar[bool] = True
    additional_speech_languages: ClassVar[frozenset[str]] = frozenset(
        (*LANG_IDS, *LANG_NAME_TO_ID)
    )
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_concurrency=2),
            next="audio_encoder",
        ),
        StageConfig(
            name="audio_encoder",
            process="pipeline",
            gpu=0,
            factory_path=f"{PACKAGE}.stages.create_audio_encoder_executor",
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            gpu=0,
            factory_path=f"{PACKAGE}.stages.create_tts_engine_executor",
            factory=FactoryArgs(
                dtype="float16", max_batch_size=1, max_batch_wait_ms=10
            ),
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            gpu=0,
            factory_path=f"{PACKAGE}.stages.create_vocoder_executor",
            factory=FactoryArgs(max_batch_size=1, max_batch_wait_ms=10),
            terminal=True,
        ),
    ]

    @model_validator(mode="after")
    def validate_single_rank(self) -> "OmniVoicePipelineConfig":
        """首版没有张量并行，拒绝将完整模型错误复制成 TP worker。"""
        if any(stage.tp_size != 1 for stage in self.stages):
            raise ValueError("OmniVoice requires tp_size=1")
        return self


EntryClass = OmniVoicePipelineConfig
