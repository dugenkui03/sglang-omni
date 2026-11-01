# SPDX-License-Identifier: Apache-2.0
"""验证合并投影的注意力等价性、检查点兼容性及实际执行次数。"""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RotaryEmbedding,
)

from sglang_omni.models.omnivoice.attention import FusedQwen3Attention
from sglang_omni.models.omnivoice.model import OmniVoiceModel


@pytest.mark.parametrize("key_value_heads", [2, 4])
@pytest.mark.parametrize("attention_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_attention_matches_separate_projections(
    key_value_heads: int, attention_bias: bool, dtype: torch.dtype
) -> None:
    """不同 Q/KV 头数、偏置和精度下保持输出，并把四次线性调用减为两次。"""
    config = Qwen3Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=key_value_heads,
        head_dim=4,
        attention_bias=attention_bias,
    )
    config._attn_implementation = "sdpa"
    with torch.random.fork_rng():
        torch.manual_seed(23)
        reference = Qwen3Attention(config, 0).to(dtype).eval()
        hidden_states = torch.randn(2, 5, 16, dtype=dtype)
    rotary = Qwen3RotaryEmbedding(config)
    embeddings = rotary(hidden_states, torch.arange(5).unsqueeze(0))
    attention_mask = torch.ones(2, 1, 5, 5, dtype=torch.bool)
    attention_mask[0, :, :, 3:] = False
    fused = FusedQwen3Attention(reference)
    parameter_count = sum(parameter.numel() for parameter in reference.parameters())
    assert sum(parameter.numel() for parameter in fused.parameters()) == parameter_count
    assert not {"q_proj", "k_proj", "v_proj"}.intersection(dict(fused.named_children()))

    with patch("torch.nn.functional.linear", wraps=torch.nn.functional.linear) as linear:
        expected, _ = reference(hidden_states, embeddings, attention_mask)
        assert linear.call_count == 4
    with patch("torch.nn.functional.linear", wraps=torch.nn.functional.linear) as linear:
        actual, _ = fused(hidden_states, embeddings, attention_mask)
        assert linear.call_count == 2
    tolerance = {
        torch.float32: 1e-6,
        torch.float16: 2e-3,
        torch.bfloat16: 2e-2,
    }[dtype]
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    # 改变补齐区域不得影响有效位置，防止合并时丢失掩码或混淆批次。
    modified = hidden_states.clone()
    modified[0, 3:] += 10
    modified_output, _ = fused(modified, embeddings, attention_mask)
    torch.testing.assert_close(modified_output[0, :3], actual[0, :3])
    torch.testing.assert_close(modified_output[1], actual[1])

    state_dict = reference.state_dict()
    exported = fused.state_dict()
    assert exported.keys() == state_dict.keys()
    for name in state_dict:
        torch.testing.assert_close(exported[name], state_dict[name], rtol=0, atol=0)
    state_dict["q_proj.weight"] = state_dict["q_proj.weight"] + 0.25
    reference.load_state_dict(state_dict)
    fused.load_state_dict(state_dict, strict=True)
    torch.testing.assert_close(
        fused(hidden_states, embeddings, attention_mask)[0],
        reference(hidden_states, embeddings, attention_mask)[0],
        rtol=tolerance,
        atol=tolerance,
    )
    incomplete = dict(state_dict)
    incomplete.pop("k_proj.weight")
    with pytest.raises(ValueError, match="Incomplete QKV"):
        fused.load_state_dict(incomplete)


@torch.inference_mode()
def test_fused_model_checkpoint_roundtrip(model: OmniVoiceModel, tmp_path: Path) -> None:
    """融合后保存仍使用官方三组参数名，恢复后可再次融合及转换 dtype。"""
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    model.fuse_qkv_projections()
    attention = model.llm.layers[0].self_attn
    model.fuse_qkv_projections()
    assert model.llm.layers[0].self_attn is attention
    model.save_pretrained(tmp_path, max_shard_size="10KB")
    restored, loading_info = OmniVoiceModel.from_pretrained(
        tmp_path, dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True
    )
    assert loading_info["missing_keys"] == set()
    assert loading_info["unexpected_keys"] == set()
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    restored.fuse_qkv_projections()
    model.to(torch.float64)
    restored.to(torch.float64)
    input_ids = torch.tensor([[[12, 13, 1, 8], [12, 13, 2, 8]]])
    audio_mask = torch.tensor([[False, False, True, True]])
    attention_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    target_positions = torch.tensor([2, 3])
    torch.testing.assert_close(
        model(input_ids, audio_mask, attention_mask, target_position_indices=target_positions),
        restored(input_ids, audio_mask, attention_mask, target_position_indices=target_positions),
        rtol=0,
        atol=0,
    )
