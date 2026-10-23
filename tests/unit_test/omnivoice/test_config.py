# SPDX-License-Identifier: Apache-2.0
"""模型发现、设备放置和检查点权重名称兼容性。"""

import inspect
from pathlib import Path

import pytest
import torch

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.omnivoice import CAPABILITIES, stages
from sglang_omni.models.omnivoice.config import EntryClass
from sglang_omni.models.omnivoice.model import OmniVoiceModel


def test_stage_graph_and_device_contract() -> None:
    """使用原生阶段，所有 GPU 工厂都接受显式设备和 GPU 编号。"""
    config = EntryClass(model_path="unused")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "audio_encoder",
        "tts_engine",
        "vocoder",
    ]
    assert {stage.process for stage in config.stages} == {"pipeline"}
    assert config.stages[-1].terminal
    assert (
        config.required_speech_reference_count == 1
        and config.speech_reference_text_required
    )
    for stage in config.stages[1:]:
        function = getattr(stages, stage.factory_path.rsplit(".", 1)[1])
        signature = inspect.signature(function)
        assert signature.parameters["device"].default is None
        assert signature.parameters["gpu_id"].default is None
    assert CAPABILITIES.supports_batch_vocoder
    assert not CAPABILITIES.supports_streaming_vocoder


def test_checkpoint_roundtrip(model, tmp_path: Path) -> None:
    """沿用官方模块名称可直接加载 safetensors，不需要权重重映射。"""
    model.save_pretrained(tmp_path)
    restored = OmniVoiceModel.from_pretrained(
        tmp_path, dtype=torch.float32, attn_implementation="sdpa"
    )
    assert restored.config.model_type == "omnivoice"
    assert not restored.llm.config.use_cache
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name])


def test_example_yaml_uses_current_config_loader() -> None:
    """启动示例必须经过真实配置解析，避免沿用已移除的 YAML 格式。"""
    config = ConfigManager.from_file("examples/configs/omnivoice.yaml").config
    assert isinstance(config, EntryClass)
    assert config.stage_named("tts_engine").factory.max_batch_size == 1
    assert config.stage_named("vocoder").terminal


def test_tensor_parallel_is_not_silently_accepted() -> None:
    """不支持的 TP 配置应在加载权重前报错。"""
    values = EntryClass(model_path="unused").model_dump()
    values["stages"][2].update(tp_size=2, gpu=[0, 1], process="generation")
    with pytest.raises(ValueError, match="tp_size=1"):
        EntryClass.model_validate(values)
