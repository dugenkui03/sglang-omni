# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xiaomi Corp. (authors: Han Zhu). See LICENSE.
"""OmniVoice 双向注意力主干及八码本输出头。"""

import torch
from torch import nn
from transformers import PreTrainedModel, Qwen3Model

from sglang_omni.models.omnivoice.hf_config import OmniVoiceConfig


class OmniVoiceModel(PreTrainedModel):
    """沿用检查点属性名称，只实现服务所需的前向计算。"""

    config_class = OmniVoiceConfig
    _supports_sdpa = True

    def __init__(self, config: OmniVoiceConfig) -> None:
        """创建 Qwen3、码本向量与音频输出头。"""
        super().__init__(config)
        config.llm_config.use_cache = False
        config.llm_config._attn_implementation = (
            "sdpa"  # noqa: leading-underscore  # Transformers 接口
        )
        self.llm = Qwen3Model(config.llm_config)
        self.audio_embeddings = nn.Embedding(
            config.num_audio_codebook * config.audio_vocab_size,
            config.llm_config.hidden_size,
        )
        self.register_buffer(
            "codebook_layer_offsets",
            torch.arange(config.num_audio_codebook) * config.audio_vocab_size,
        )
        self.audio_heads = nn.Linear(
            config.llm_config.hidden_size,
            config.num_audio_codebook * config.audio_vocab_size,
            bias=False,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        """为检查点加载和前向计算提供文本向量表。"""
        return self.llm.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """将 [B,C,S] 输入转换为 [B,C,S,V] 音频 logits。"""
        text_embeddings = self.llm.embed_tokens(input_ids[:, 0, :])
        audio_ids = input_ids * audio_mask.unsqueeze(
            1
        ) + self.codebook_layer_offsets.view(1, -1, 1)
        audio_embeddings = self.audio_embeddings(audio_ids).sum(dim=1)
        embeddings = torch.where(
            audio_mask.unsqueeze(-1), audio_embeddings, text_embeddings
        )
        hidden_states = self.llm(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        logits = self.audio_heads(hidden_states)
        return logits.view(
            input_ids.shape[0],
            input_ids.shape[-1],
            self.config.num_audio_codebook,
            self.config.audio_vocab_size,
        ).permute(0, 2, 1, 3)
