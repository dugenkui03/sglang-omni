# SPDX-License-Identifier: Apache-2.0
"""加载权重后就地启用 RMSNorm 融合，仅用于推理。"""

from types import MethodType

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

if torch.version.cuda is not None:
    from flashinfer.norm import rmsnorm


def fuse_rmsnorm(normalization: Qwen3RMSNorm) -> None:
    """保留原归一化对象、权重及 eps；重复调用不再次包装 forward。"""
    if normalization.forward.__func__ is fused_rmsnorm_forward:
        return
    normalization.forward = MethodType(fused_rmsnorm_forward, normalization)


def fused_rmsnorm_forward(
    self: Qwen3RMSNorm, hidden_states: torch.Tensor
) -> torch.Tensor:
    """CUDA FP16/BF16 且权重精度相同时使用融合内核，其余沿用原计算。"""
    if (
        torch.version.cuda is not None
        and hidden_states.is_cuda
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and self.weight.dtype == hidden_states.dtype
    ):
        shape = hidden_states.shape
        # 只沿最后一维归一化；展平前置维并处理 Q/K 切片的不连续布局。
        normalized_states = rmsnorm(
            hidden_states.reshape(-1, shape[-1]).contiguous(),
            self.weight,
            eps=self.variance_epsilon,
        )
        return normalized_states.view(shape)
    return Qwen3RMSNorm.forward(self, hidden_states)
