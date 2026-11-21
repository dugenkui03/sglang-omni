# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
"""在推理中融合 Attention 后的残差加法与 RMSNorm。"""

from types import MethodType
from typing import Unpack

import torch
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from transformers.utils.generic import TransformersKwargs

if torch.version.cuda is not None:
    # tip 出现如下模式计算的时候，可以使用 fused_add_rmsnorm 进行优化
    #   summed = tensor_a + tensor_b
    #   normed = rmsnorm(summed)
    from flashinfer.norm import fused_add_rmsnorm
else:
    fused_add_rmsnorm = None


def fuse_residual_add_rmsnorm(layer: Qwen3DecoderLayer) -> None:
    """替换整个 Transformer 层的 forward，融合其中的 Attention 残差加法与 RMSNorm。"""
    forward = layer.forward
    layer.forward = (
        forward
        if forward.__func__ is fused_decoder_forward
        else MethodType(fused_decoder_forward, layer)
    )


def fused_decoder_forward(
    self: Qwen3DecoderLayer,  # tip 原来的 transformer 层
    hidden_states: torch.Tensor,  # 当前层的输入
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    use_cache: bool | None = False,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> torch.Tensor:
    """替代 Qwen3DecoderLayer.forward，沿用 Attention 和 MLP 两个子层的归一化 → 核心计算 → 残差相加顺序。

    将 Attention 的残差相加与 MLP 前的归一化融合。CUDA 路径原地更新本轮临时缓冲区，仅用于内部推理。
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    if (
        fused_add_rmsnorm is not None  # CUDA 构建已导入融合算子
        and hidden_states.is_cuda  # 如果是 cuda 设备
        and hidden_states.dtype in (torch.float16, torch.bfloat16)  # 精度限制
        and residual.dtype == hidden_states.dtype  # 数据类型限制
        and self.post_attention_layernorm.weight.dtype
        == hidden_states.dtype  # 数据类型限制
    ):
        # [batch_size * 2, max_sequence_length, hidden_size]
        shape = hidden_states.shape
        # [batch_size * 2 * max_sequence_length, hidden_size]
        residual = residual.reshape(-1, shape[-1]).contiguous()
        hidden_states = hidden_states.reshape(-1, shape[-1]).contiguous()

        # note rmsnorm( tensor_1 + tensor_2 ) 计算融合成一次计算
        # 原地写回：residual 保存相加结果，hidden_states 保存归一化结果。
        fused_add_rmsnorm(
            hidden_states,  # attn 的结果，没有残差相加 // tensor_1
            residual,  # attn 待相加的残差 // tensor_2
            self.post_attention_layernorm.weight,  # 对 tensor_1 + tensor_2 相加结果进行归一化的权重
            eps=self.post_attention_layernorm.variance_epsilon,
        )
        residual = residual.view(shape)
        hidden_states = hidden_states.view(shape)
    else:
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    # MLP 后的残差加法沿用原计算，保持各层之间的 Tensor 返回协议。
    return residual + hidden_states
