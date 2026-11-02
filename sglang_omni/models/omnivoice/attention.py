# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
"""加载后就地融合 QKV，仅用于推理，不转换检查点保存格式。"""

from types import MethodType
from typing import Unpack

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


def fuse_qkv_attention(attention: Qwen3Attention) -> None:
    """在原注意力对象上拼接权重、替换 forward；重复调用不重新融合。"""
    if attention.forward.__func__ is fused_attention_forward:
        return
    projections = (attention.q_proj, attention.k_proj, attention.v_proj)
    attention.qkv_sizes = tuple(layer.out_features for layer in projections)
    # W_Q、W_K、W_V 按行拼接，只在初始化时执行一次。
    attention.qkv_weight = nn.Parameter(
        torch.cat([layer.weight.detach() for layer in projections]), requires_grad=False
    )
    attention.qkv_bias = None
    if attention.q_proj.bias is not None:
        attention.qkv_bias = nn.Parameter(
            torch.cat([layer.bias.detach() for layer in projections]), requires_grad=False
        )
    # 释放原来的三组投影，避免重复占用 GPU 显存；其他配置和层直接复用。
    del attention.q_proj, attention.k_proj, attention.v_proj
    attention.forward = MethodType(fused_attention_forward, attention)


def fused_attention_forward(
    self: Qwen3Attention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """W_Q、W_K、W_V 合并后一次计算，再拆分 Q/K/V；后续沿用原注意力逻辑。"""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    # 核心优化：一次投影，再按各自宽度拆分（Q 与 K/V 可以不等宽）。
    qkv = F.linear(hidden_states, self.qkv_weight, self.qkv_bias)
    query_states, key_states, value_states = qkv.split(self.qkv_sizes, dim=-1)

    query_states = self.q_norm(query_states.reshape(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(key_states.reshape(hidden_shape)).transpose(1, 2)
    value_states = value_states.reshape(hidden_shape).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, *position_embeddings
    )
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, sliding_window=self.sliding_window, **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights
