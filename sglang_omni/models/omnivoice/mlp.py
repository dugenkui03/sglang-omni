# SPDX-License-Identifier: Apache-2.0
"""合并 Gate/Up 投影，并在 CUDA 上融合 SiLU 与逐元素乘法，仅用于推理。"""

from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

if torch.version.cuda is not None:
    from flashinfer.activation import silu_and_mul


def fuse_gate_up_mlp(mlp: Qwen3MLP) -> None:
    """
    Qwen3MLP 的 mlp 实现： down_proj = self.down_proj(
                                                         self.act_fn(           # 激活函数
                                                             self.gate_proj(x) # gate
                                                         )
                                                         * self.up_proj(x)。  # up
                                                   )

    """
    if mlp.forward.__func__ is fused_gate_up_forward:
        return
    # Qwen3 的 Gate 和 Up 都把 hidden_size 投影到 intermediate_size
    # note 拼接 gate_proj 和 up_prob 的 weight
    mlp.gate_up_weight = nn.Parameter(
        torch.cat([mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach()], dim=0),
        requires_grad=False,
    )
    # 原权重已合并，释放旧投影；保留原来的激活函数和 down_proj。
    del mlp.gate_proj, mlp.up_proj
    mlp.forward = MethodType(fused_gate_up_forward, mlp)


def fused_gate_up_forward(self: Qwen3MLP, hidden_states: torch.Tensor) -> torch.Tensor:
    """CUDA FP16/BF16 使用融合 SiLU/Mul，其余设备、精度或激活保留原计算。"""
    # 输出最后一维是 2 * intermediate_size，前半是 Gate，后半是 Up。
    # NOTE F.linear 逻辑是
    #       gate_up_states = hidden_states @ self.gate_up_weight.T // 和后者的转置相乘
    gate_up_states = F.linear(hidden_states, self.gate_up_weight)
    if (
        gate_up_states.is_cuda
        and gate_up_states.dtype in (torch.float16, torch.bfloat16)
        and self.config.hidden_act == "silu"
    ):
        # 一次 CUDA kernel 完成 SiLU(Gate) * Up，避免写回独立的激活中间结果。
        activated_states = silu_and_mul(gate_up_states)
    else:
        gate_states, up_states = gate_up_states.chunk(2, dim=-1)
        activated_states = self.act_fn(gate_states) * up_states
    return self.down_proj(activated_states)
