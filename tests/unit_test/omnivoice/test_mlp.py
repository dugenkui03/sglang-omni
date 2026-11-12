# SPDX-License-Identifier: Apache-2.0
"""验证 Gate/Up 合并及 CUDA SiLU/Mul 融合与原 MLP 的数值一致性。"""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

import sglang_omni.models.omnivoice.mlp as mlp_module
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
    with patch(
        "torch.nn.functional.linear", wraps=torch.nn.functional.linear
    ) as linear:
        expected = reference(hidden_states)
        assert linear.call_count == 3
    with patch(
        "torch.nn.functional.linear", wraps=torch.nn.functional.linear
    ) as linear:
        actual = fused(hidden_states)
        assert linear.call_count == 2
    tolerance = {
        torch.float32: 1e-6,
        torch.float16: 2e-3,
        torch.bfloat16: 2e-2,
    }[dtype]
    assert actual.shape == hidden_states.shape
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU 和 FlashInfer")
@pytest.mark.parametrize(
    ("dtype", "hidden_act"),
    [
        (torch.float16, "silu"),
        (torch.bfloat16, "silu"),
        (torch.float32, "silu"),
        (torch.float16, "gelu"),
    ],
)
@torch.inference_mode()
def test_cuda_silu_mul_matches_original_mlp(
    dtype: torch.dtype, hidden_act: str
) -> None:
    """真实融合内核保留批次和序列形状；FP32、非 SiLU 不应进入该内核。"""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("当前 GPU 不支持 BF16")
    config = Qwen3Config(hidden_size=16, intermediate_size=32, hidden_act=hidden_act)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(29)
        reference = Qwen3MLP(config).to(device="cuda", dtype=dtype).eval()
        hidden_states = torch.randn(2, 3, 16, device="cuda", dtype=dtype)
    fused = deepcopy(reference)
    fuse_gate_up_mlp(fused)
    expected = reference(hidden_states)
    with patch.object(
        mlp_module, "silu_and_mul", wraps=mlp_module.silu_and_mul
    ) as activation:
        actual = fused(hidden_states)
        assert activation.call_count == int(
            dtype in (torch.float16, torch.bfloat16) and hidden_act == "silu"
        )
    assert actual.shape == hidden_states.shape
    assert actual.dtype == dtype
    assert actual.device == hidden_states.device
    tolerance = {
        torch.float32: 1e-6,
        torch.float16: 2e-3,
        torch.bfloat16: 2e-2,
    }[dtype]
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
