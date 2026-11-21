# SPDX-License-Identifier: Apache-2.0
"""验证 RMSNorm 融合保留归一化轴、参数及原精度计算行为。"""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

import sglang_omni.models.omnivoice.norm as norm_module
from sglang_omni.models.omnivoice.model import OmniVoiceModel
from sglang_omni.models.omnivoice.norm import fuse_rmsnorm, fused_rmsnorm_forward


@pytest.mark.parametrize("shape", [(2, 16), (2, 3, 16), (2, 3, 2, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_cpu_rmsnorm_preserves_original(
    shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    """CPU 原路径在不同维数和不连续输入下逐值一致，重复启用不改变参数。"""
    with torch.random.fork_rng():
        torch.manual_seed(37)
        reference = Qwen3RMSNorm(shape[-1], eps=1e-4).to(dtype).eval()
        reference.weight.uniform_(0.5, 1.5)
        hidden_states = torch.randn(*shape[:-1], shape[-1] * 2, dtype=dtype)[..., ::2]
    fused = deepcopy(reference)
    weight = fused.weight
    fuse_rmsnorm(fused)
    forward = fused.forward
    fuse_rmsnorm(fused)
    assert fused.forward is forward
    assert fused.weight is weight
    assert fused.variance_epsilon == reference.variance_epsilon
    original_input = hidden_states.clone()
    expected = reference(hidden_states)
    actual = fused(hidden_states)
    assert actual.shape == hidden_states.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden_states, original_input, rtol=0, atol=0)


def test_model_covers_all_rmsnorms(model: OmniVoiceModel) -> None:
    """启用入口必须包含主干、Q/K 和最终归一化，并保留已加载的参数对象。"""
    normalizations = [model.llm.norm]
    for layer in model.llm.layers:
        normalizations.extend(
            [
                layer.input_layernorm,
                layer.post_attention_layernorm,
                layer.self_attn.q_norm,
                layer.self_attn.k_norm,
            ]
        )
    parameters = dict(model.named_parameters())
    model.fuse_rmsnorm()
    for normalization in normalizations:
        assert normalization.forward.__func__ is fused_rmsnorm_forward
    assert parameters.keys() == dict(model.named_parameters()).keys()
    for name, parameter in model.named_parameters():
        assert parameter is parameters[name]


@pytest.mark.accelerator
@pytest.mark.skipif(
    torch.version.cuda is None or not torch.cuda.is_available(),
    reason="需要 NVIDIA CUDA GPU 和 FlashInfer",
)
@pytest.mark.parametrize("shape", [(2, 16), (2, 3, 16), (2, 3, 2, 8)])
@pytest.mark.parametrize(
    ("dtype", "weight_dtype"),
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float16, torch.float32),
    ],
)
@torch.inference_mode()
def test_cuda_rmsnorm_matches_original(
    shape: tuple[int, ...], dtype: torch.dtype, weight_dtype: torch.dtype
) -> None:
    """真实内核处理不连续的 2D/3D/4D 输入；FP32 和混合精度保留原计算。"""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("当前 GPU 不支持 BF16")
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(37)
        reference = (
            Qwen3RMSNorm(shape[-1], eps=1e-4)
            .to(device="cuda", dtype=weight_dtype)
            .eval()
        )
        reference.weight.uniform_(0.5, 1.5)
        hidden_states = torch.randn(
            *shape[:-1], shape[-1] * 2, device="cuda", dtype=dtype
        )[..., ::2]
    fused = deepcopy(reference)
    fuse_rmsnorm(fused)
    original_input = hidden_states.clone()
    expected = reference(hidden_states)
    with patch.object(norm_module, "rmsnorm", wraps=norm_module.rmsnorm) as kernel:
        actual = fused(hidden_states)
        assert kernel.call_count == int(
            dtype in (torch.float16, torch.bfloat16) and weight_dtype == dtype
        )
    assert actual.shape == hidden_states.shape
    assert actual.dtype == expected.dtype
    assert actual.device == hidden_states.device
    tolerance = {
        torch.float32: 0,
        torch.float16: 2e-3,
        torch.bfloat16: 2e-2,
    }[dtype]
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(hidden_states, original_input, rtol=0, atol=0)
