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
        # 已经处理过，则直接返回
        return
    # tip 创建方法、这个方法实现是 fused_rmsnorm_forward、第一个参数 self 是normalization、赋值给 normalization.forward
    normalization.forward = MethodType(fused_rmsnorm_forward, normalization)


def fused_rmsnorm_forward(
    self: Qwen3RMSNorm,
    hidden_states: torch.Tensor
) -> torch.Tensor:
    """
       Root Mean Square Normalization，均方根归一化
        Square：平方
        Mean：平方平均
        Root：求开方
        Normalization：归一化；与 RMS 相除进行归一化。
    """
    if (
        torch.version.cuda is not None
        and hidden_states.is_cuda
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and self.weight.dtype == hidden_states.dtype
    ):
        shape = hidden_states.shape
        # note 最后一个序列是一个token对应的向量，每个token内部进行单独的归一化
        #   output = [x / sqrt(mean(x ** 2) + eps)] * weight。注意，每一个元素都与 sqrt(mean(x ** 2) + eps) 做除法计算
        normalized_states = rmsnorm(
            hidden_states.reshape(-1, shape[-1]).contiguous(), # # 合并前置维、保留最后一维，每行向量独立计算 RMS
            self.weight, # 训练出的缩放向量，与归一化结果逐元素相乘，对一个token向量的不同元素进行不同程度的缩放（✖️计算）
            eps=self.variance_epsilon, # epsilon 极小值、比如0.000001，避免分母为0
        )
        return normalized_states.view(shape)
    return Qwen3RMSNorm.forward(self, hidden_states)
