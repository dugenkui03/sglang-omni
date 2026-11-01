# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
"""合并 Qwen3 的 QKV 投影，保持注意力计算与官方检查点格式。"""

from typing import Unpack

import torch
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
        """复用原有归一化与输出投影，只替换 QKV 线性计算。"""
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
        self.qkv_proj = nn.Linear(
            attention.q_proj.in_features,
            sum(self.projection_sizes),
            bias=attention.q_proj.bias is not None,
            device="meta",
            dtype=attention.q_proj.weight.dtype,
        )
        self.qkv_proj.weight = nn.Parameter(
            torch.cat([layer.weight.detach() for layer in projections], dim=0),
            requires_grad=attention.q_proj.weight.requires_grad,
        )
        if attention.q_proj.bias is not None:
            self.qkv_proj.bias = nn.Parameter(
                torch.cat([layer.bias.detach() for layer in projections], dim=0),
                requires_grad=attention.q_proj.bias.requires_grad,
            )
        self.train(attention.training)
        self.register_state_dict_post_hook(export_qkv_projections)
        self.register_load_state_dict_pre_hook(import_qkv_projections)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """一次投影后按 Q/K/V 实际宽度拆分，其余保持 Qwen3 的计算顺序。"""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, key_states, value_states = self.qkv_proj(hidden_states).split(
            self.projection_sizes, dim=-1
        )
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


def export_qkv_projections(
    module: FusedQwen3Attention,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict[str, int],
) -> None:
    """保存时恢复原 Q/K/V 名称与宽度；metadata 由 PyTorch 回调协议传入。"""
    for parameter_name, parameter in module.qkv_proj.named_parameters():
        fused = state_dict.pop(f"{prefix}qkv_proj.{parameter_name}")
        for projection_name, tensor in zip(
            ("q_proj", "k_proj", "v_proj"),
            fused.split(module.projection_sizes, dim=0),
            strict=True,
        ):
            state_dict[f"{prefix}{projection_name}.{parameter_name}"] = tensor.clone()


def import_qkv_projections(
    module: FusedQwen3Attention,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict[str, int],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    """重新加载时合并官方 Q/K/V；其余参数由 PyTorch 加载回调协议提供。"""
    for parameter_name, parameter in module.qkv_proj.named_parameters():
        names = [
            f"{prefix}{projection_name}.{parameter_name}"
            for projection_name in ("q_proj", "k_proj", "v_proj")
        ]
        if any(name in state_dict for name in names):
            if not all(name in state_dict for name in names):
                raise ValueError(f"Incomplete QKV parameters: {names}")
            for name, width in zip(names, module.projection_sizes, strict=True):
                if state_dict[name].shape != (width, *parameter.shape[1:]):
                    raise ValueError(f"Invalid QKV parameter shape: {name}")
            state_dict[f"{prefix}qkv_proj.{parameter_name}"] = torch.cat(
                [state_dict.pop(name) for name in names], dim=0
            )
