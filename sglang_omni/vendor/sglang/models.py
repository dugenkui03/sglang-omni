"""Vendor wrapper for sglang.srt.models.utils.

Centralize third-party imports and apply monkey patches here.

Patch applied to apply_qk_norm:
  - Skip fused_inplace_qknorm when q_norm.cast_x_before_out_mul is True,
    ensuring HF-compatible RMSNorm cast order for QK normalization.
This patch can be removed once upstream SGLang merges the equivalent change.
"""

from __future__ import annotations

from typing import Optional, Tuple

import sglang.srt.models.utils as _sglang_models_utils
import torch
from sglang.kernels.ops.attention.rope import FusedSetKVBufferArg
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.utils import (
    create_fused_set_kv_buffer_arg,
    enable_fused_set_kv_buffer,
)

# ---------------------------------------------------------------------------
# apply_qk_norm monkey-patch: skip fused QK norm when cast_x_before_out_mul
# ---------------------------------------------------------------------------
_orig_apply_qk_norm = _sglang_models_utils.apply_qk_norm


def apply_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    head_dim: int,
    alt_stream: Optional[torch.cuda.Stream] = None,
    allow_inplace: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """对注意力层的 Query(q) 和 Key(k) 按注意力头(head_dim)做 RMSNorm 归一化(QK-Norm)。

    参数:
        q: Query 张量，形状 [num_tokens, num_heads * head_dim]
        k: Key 张量，形状 [num_tokens, num_kv_heads * head_dim]
        q_norm / k_norm: Q 和 K 各自对应的 RMSNorm 模块（权重形状均为 [head_dim]）
        head_dim: 单个注意力头的维度大小，归一化在每个 head 内部独立计算
        alt_stream: 可选的辅助 CUDA Stream，用于让 q_norm 和 k_norm 并行执行
        allow_inplace: 是否允许原地(in-place)融合算子直接修改 q、k 显存

    返回值:
        归一化后的 (q, k) 元组，形状与输入保持一致。
    """
    # 当要求在乘权重前先把 x 转回低精度（对齐 HuggingFace 精度行为）时，
    # 禁用底层融合原地算子，走标准 RMSNorm 分支以保证数值一致
    if q_norm.cast_x_before_out_mul:
        allow_inplace = False
    return _orig_apply_qk_norm(
        q,
        k,
        q_norm,
        k_norm,
        head_dim,
        alt_stream=alt_stream,
        allow_inplace=allow_inplace,
    )


# Patch the source module so any direct imports also see the change.
_sglang_models_utils.apply_qk_norm = apply_qk_norm

__all__ = [
    "FusedSetKVBufferArg",
    "apply_qk_norm",
    "create_fused_set_kv_buffer_arg",
    "enable_fused_set_kv_buffer",
]
