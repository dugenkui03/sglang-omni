# SPDX-License-Identifier: Apache-2.0
"""验证合并投影的注意力等价性及实际执行次数。"""

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


@torch.inference_mode()
def test_fusion_is_idempotent(model: OmniVoiceModel) -> None:
    """重复启用不重新拼接权重，保留已创建的注意力对象。"""
    attention = model.llm.layers[0].self_attn
    weight = attention.qkv_weight
    model.fuse_qkv_projections()
    assert model.llm.layers[0].self_attn is attention
    assert attention.qkv_weight is weight
