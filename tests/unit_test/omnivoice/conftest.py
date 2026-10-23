# SPDX-License-Identifier: Apache-2.0
"""无需下载权重的微型模型与 tokenizer。"""

import pytest
import torch
from transformers import BatchEncoding, Qwen3Config

from sglang_omni.models.omnivoice.hf_config import OmniVoiceConfig
from sglang_omni.models.omnivoice.model import OmniVoiceModel


class TinyTokenizer:
    """用稳定的小词表模拟文本接口，避免网络依赖。"""

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> BatchEncoding:
        """按字符产生固定 token；测试仅关心分段与张量布局。"""
        values = [12 + ord(character) % 80 for character in text]
        return BatchEncoding(
            {"input_ids": torch.tensor([values]) if return_tensors else values}
        )


@pytest.fixture
def tokenizer() -> TinyTokenizer:
    """为算法测试提供无状态 tokenizer。"""
    return TinyTokenizer()


@pytest.fixture
def model() -> OmniVoiceModel:
    """构造真实 Qwen3 微型网络，执行真实双向注意力。"""
    with torch.random.fork_rng():
        torch.manual_seed(7)
        config = OmniVoiceConfig(
            llm_config=Qwen3Config(
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                vocab_size=128,
                max_position_embeddings=512,
            ),
            num_audio_codebook=2,
            audio_vocab_size=9,
            audio_mask_id=8,
        )
        return OmniVoiceModel(config).eval()
