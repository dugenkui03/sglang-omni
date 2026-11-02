# SPDX-License-Identifier: Apache-2.0
"""验证 Gate/Up 合并后的数值、线性调用次数和重复启用行为。"""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from sglang_omni.models.omnivoice.mlp import fuse_gate_up_mlp


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_gate_up_matches_separate_projections(dtype: torch.dtype) -> None:
    """保留 SwiGLU 输出及原模块，把三次线性调用变为两次且不复制保留旧权重。"""
    config = Qwen3Config(hidden_size=16, intermediate_size=32, hidden_act="silu")
    with torch.random.fork_rng():
        torch.manual_seed(29)
        reference = Qwen3MLP(config).to(dtype).eval()
        hidden_states = torch.randn(2, 3, 16, dtype=dtype)
    fused = deepcopy(reference)
    down_projection, activation = fused.down_proj, fused.act_fn
    fuse_gate_up_mlp(fused)
    assert fused.down_proj is down_projection
    assert fused.act_fn is activation
    assert sum(parameter.numel() for parameter in fused.parameters()) == sum(
        parameter.numel() for parameter in reference.parameters()
    )
    assert not {"gate_proj", "up_proj"}.intersection(dict(fused.named_children()))
    weight = fused.gate_up_weight
    fuse_gate_up_mlp(fused)
    assert fused.gate_up_weight is weight
    with patch("torch.nn.functional.linear", wraps=torch.nn.functional.linear) as linear:
        expected = reference(hidden_states)
        assert linear.call_count == 3
    with patch("torch.nn.functional.linear", wraps=torch.nn.functional.linear) as linear:
        actual = fused(hidden_states)
        assert linear.call_count == 2
    tolerance = {
        torch.float32: 1e-6,
        torch.float16: 2e-3,
        torch.bfloat16: 2e-2,
    }[dtype]
    assert actual.shape == hidden_states.shape
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
