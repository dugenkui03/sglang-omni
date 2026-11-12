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
    from flashinfer.norm import fused_add_rmsnorm


def fuse_residual_add_rmsnorm(layer: Qwen3DecoderLayer) -> None:
    """保留原 DecoderLayer 和参数，就地绑定推理用的 forward。"""
    if layer.forward.__func__ is fused_decoder_forward:
        return
    layer.forward = MethodType(fused_decoder_forward, layer)


def fused_decoder_forward(
    self: Qwen3DecoderLayer,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    use_cache: bool | None = False,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> torch.Tensor:
    """融合 Attention 后的 Add/Norm；CUDA 路径会更新传入的临时 hidden_states。

    仅用于模型内部推理；这些张量来自本轮独立 Embedding 或上一层输出。
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
        torch.version.cuda is not None
        and hidden_states.is_cuda
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and residual.dtype == hidden_states.dtype
        and self.post_attention_layernorm.weight.dtype == hidden_states.dtype
    ):
        shape = hidden_states.shape
        residual = residual.reshape(-1, shape[-1]).contiguous()
        hidden_states = hidden_states.reshape(-1, shape[-1]).contiguous()
        # 一次内核：residual 写入相加结果，hidden_states 写入归一化结果。
        fused_add_rmsnorm(
            hidden_states,
            residual,
            self.post_attention_layernorm.weight,
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
