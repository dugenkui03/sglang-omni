# SPDX-License-Identifier: Apache-2.0
"""验证残差融合保留模型结果、参数和多轮复用的固定输入。"""

from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

import sglang_omni.models.omnivoice.decoder as decoder_module
from sglang_omni.models.omnivoice.decoder import fused_decoder_forward
from sglang_omni.models.omnivoice.generation import pack_batch
from sglang_omni.models.omnivoice.model import OmniVoiceModel


def test_residual_fusion_is_idempotent(model: OmniVoiceModel) -> None:
    """重复启用保留所有层、参数和已绑定方法，不新增模型权重。"""
    layers = list(model.llm.layers)
    forwards = [layer.forward for layer in layers]
    parameters = dict(model.named_parameters())
    model.fuse_residual_add_rmsnorm()
    for index, layer in enumerate(model.llm.layers):
        assert layer is layers[index]
        assert layer.forward is forwards[index]
        assert layer.forward.__func__ is fused_decoder_forward
    assert parameters.keys() == dict(model.named_parameters()).keys()
    for name, parameter in model.named_parameters():
        assert parameter is parameters[name]


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", torch.float32),
        ("cpu", torch.float16),
        ("cpu", torch.bfloat16),
        *[
            pytest.param(
                "cuda",
                dtype,
                marks=[
                    pytest.mark.accelerator,
                    pytest.mark.skipif(
                        torch.version.cuda is None or not torch.cuda.is_available(),
                        reason="需要 NVIDIA CUDA GPU 和 FlashInfer",
                    ),
                ],
            )
            for dtype in (torch.float16, torch.bfloat16, torch.float32)
        ],
    ],
)
@torch.inference_mode()
def test_residual_fusion_matches_model_across_steps(
    model: OmniVoiceModel, device: str, dtype: torch.dtype
) -> None:
    """两层模型在变长 CFG 批次、多轮更新下对齐，原地计算不得污染固定输入。"""
    if (
        device == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("当前 GPU 不支持 BF16")
    config = deepcopy(model.config)
    config.llm_config.num_hidden_layers = 2
    config.llm_config.layer_types = ["full_attention"] * 2
    with torch.random.fork_rng():
        torch.manual_seed(41)
        reference = OmniVoiceModel(config).to(device=device, dtype=dtype).eval()
    reference.fuse_qkv_projections()
    reference.fuse_gate_up_projections()
    reference.fuse_rmsnorm()
    fused = deepcopy(reference)
    fused.fuse_residual_add_rmsnorm()
    inputs = [
        (
            torch.tensor([[1, 2, 3, 8, 8], [1, 2, 4, 8, 8]], device=device),
            torch.tensor([False, False, True, True, True], device=device),
        ),
        (
            torch.tensor([[2, 3, 4, 8], [2, 4, 5, 8]], device=device),
            torch.tensor([False, True, True, True], device=device),
        ),
    ]
    input_ids, audio_mask, attention_mask = pack_batch(inputs, [2, 1], 8)
    target_indices = torch.tensor([3, 4, 8, 10, 11, 15], device=device)
    fixed_inputs = reference.prepare_fixed_inputs(input_ids, audio_mask)
    snapshots = {name: tensor.clone() for name, tensor in vars(fixed_inputs).items()}
    kernel_context = (
        patch.object(
            decoder_module,
            "fused_add_rmsnorm",
            wraps=decoder_module.fused_add_rmsnorm,
        )
        if device == "cuda"
        else nullcontext()
    )
    tolerance = (
        {torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
        if device == "cuda" and dtype != torch.float32
        else 0
    )
    with kernel_context as kernel:
        for step in range(2):
            input_ids[0, :, 3:5] = step
            input_ids[1, :, 3:4] = step
            input_ids[2, :, :2] = step
            input_ids[3, :, :1] = step
            original_ids = input_ids.clone()
            expected = reference(
                input_ids,
                fixed_inputs,
                attention_mask,
                target_position_indices=target_indices,
            )
            actual = fused(
                input_ids,
                fixed_inputs,
                attention_mask,
                target_position_indices=target_indices,
            )
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(input_ids, original_ids, rtol=0, atol=0)
            for name, tensor in vars(fixed_inputs).items():
                torch.testing.assert_close(tensor, snapshots[name], rtol=0, atol=0)
        if device == "cuda":
            assert kernel.call_count == (4 if dtype != torch.float32 else 0)
