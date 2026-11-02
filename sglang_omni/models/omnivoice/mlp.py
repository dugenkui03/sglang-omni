# SPDX-License-Identifier: Apache-2.0
"""加载权重后就地合并 Gate/Up 投影，仅用于推理。"""

from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP


def fuse_gate_up_mlp(mlp: Qwen3MLP) -> None:
    """拼接两个无偏置投影的权重并绑定 forward，重复调用不重新融合。"""
    if mlp.forward.__func__ is fused_gate_up_forward:
        return
    # Qwen3 的 Gate 和 Up 都把 hidden_size 投影到 intermediate_size。
    mlp.gate_up_weight = nn.Parameter(
        torch.cat([mlp.gate_proj.weight.detach(), mlp.up_proj.weight.detach()], dim=0),
        requires_grad=False,
    )
    # 原权重已合并，释放旧投影；保留原来的激活函数和 down_proj。
    del mlp.gate_proj, mlp.up_proj
    mlp.forward = MethodType(fused_gate_up_forward, mlp)


def fused_gate_up_forward(self: Qwen3MLP, hidden_states: torch.Tensor) -> torch.Tensor:
    """一次线性计算得到 Gate/Up，再执行原来的 SiLU、逐元素乘法和降维。"""
    # 输出最后一维是 2 * intermediate_size，前半是 Gate，后半是 Up。
    gate_up_states = F.linear(hidden_states, self.gate_up_weight)
    gate_states, up_states = gate_up_states.chunk(2, dim=-1)
    return self.down_proj(self.act_fn(gate_states) * up_states)
