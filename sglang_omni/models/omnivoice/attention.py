# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
"""加载权重后合并 QKV，仅用于推理，不转换检查点保存格式。"""

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


class FusedQwen3Attention(Qwen3Attention):
    """将已加载的三组权重一次性拼接，不保留重复的 QKV 参数。"""

    def __init__(self, attention: Qwen3Attention) -> None:
        """
           NOTE 使用原来的 Qwen3Attention 创建 FusedQwen3Attention
        """
        nn.Module.__init__(self)
        self.config = attention.config
        self.layer_idx = attention.layer_idx
        self.layer_type = attention.layer_type
        self.head_dim = attention.head_dim
        self.num_key_value_groups = attention.num_key_value_groups
        self.scaling = attention.scaling
        self.attention_dropout = attention.attention_dropout
        self.is_causal = attention.is_causal
        self.sliding_window = attention.sliding_window
        self.q_norm = attention.q_norm
        self.k_norm = attention.k_norm
        self.o_proj = attention.o_proj
        projections = (attention.q_proj, attention.k_proj, attention.v_proj)
        self.projection_sizes = tuple(layer.out_features for layer in projections)

        # 将 W_Q、W_K、W_V 按行拼接；初始化时只做一次。
        self.qkv_weight = nn.Parameter(
            torch.cat([layer.weight.detach() for layer in projections], dim=0),
            requires_grad=attention.q_proj.weight.requires_grad,
        )
        self.qkv_bias = None
        if attention.q_proj.bias is not None:
            self.qkv_bias = nn.Parameter(
                torch.cat([layer.bias.detach() for layer in projections], dim=0),
                requires_grad=attention.q_proj.bias.requires_grad,
            )
        self.train(attention.training)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
           NOTE W_Q， W_K，W_V 拼接在一块，一次tensor计算获取一个输入向量的 qkv(拼接在一块的)，然后做拆分
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # 一次矩阵乘法算出 QKV，再按各自宽度拆开；GQA 的 Q 与 K/V 宽度可以不同。
        qkv = F.linear(hidden_states, self.qkv_weight, self.qkv_bias)
        query_states, key_states, value_states = qkv.split(self.projection_sizes, dim=-1)

        # 以下沿用 Qwen3 原有的注意力计算。
        query_states = self.q_norm(query_states.reshape(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key_states.reshape(hidden_shape)).transpose(1, 2)
        value_states = value_states.reshape(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights
