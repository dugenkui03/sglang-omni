# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xiaomi Corp. (authors: Han Zhu). See LICENSE.
"""OmniVoice 双向注意力主干及八码本输出头。"""

import torch
from torch import nn
from transformers import PreTrainedModel, Qwen3Model

from sglang_omni.models.omnivoice.hf_config import OmniVoiceConfig


class OmniVoiceModel(PreTrainedModel):

    config_class = OmniVoiceConfig
    _supports_sdpa = True

    def __init__(self, config: OmniVoiceConfig) -> None:
        """创建 Qwen3、码本向量与音频输出头。"""
        super().__init__(config)
        config.llm_config.use_cache = False
        config.llm_config._attn_implementation = (
            "sdpa"  # noqa: leading-underscore  # Transformers 接口
        )
        # NOTE 文本模型赋值，embed_tokens 是 Qwen3Model 的属性、用来文本向量化
        self.llm = Qwen3Model(config.llm_config)

        # NOTE 音频向量化
        self.audio_embeddings = nn.Embedding(
            config.num_audio_codebook * config.audio_vocab_size,
            config.llm_config.hidden_size,
        )
        self.register_buffer(
            # 值是 [0, 1025, 2050, 3075, 4100, 5125, 6150, 7175]
            #   背景，码本编号共 vocab_size 个，但是相同码本在不同层、含义也不一样，所以码本对应的向量是不同的，音频码本对应的向量数量是 8 * vocab_size 个
            "codebook_layer_offsets",
            # 初始化 tensor、元素从0开始、共 num_audio_codebook 个元素
            torch.arange(config.num_audio_codebook)
                # 每个元素和 vocab_size 相乘
                * config.audio_vocab_size,
        )

        # NOTE: 定义一个线性层
        self.audio_heads = nn.Linear(
            config.llm_config.hidden_size, # input dim
            # NOTE
            #   输出维度是 audio_vocab_size * 8 是将 transformer layer 输出的 hidden_state 转换成 8 层 audio_codes
            #   每一个 audio_vocab_size 选出一个 audio code
            config.num_audio_codebook * config.audio_vocab_size,
            bias=False,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        # tip self.llm 初始化的时候赋值 Qwen3TTS 模型， embed_tokens 是这个字段的固定属性
        return self.llm.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor, # [batch_size * 2, 8, max_len]
        audio_mask: torch.Tensor, # [batch_size * 2, max_len]
        attention_mask: torch.Tensor, # [batch_size * 2, 1, max_len, max_len] TODO 1 对应助理李头的维度 // 不懂
    ) -> torch.Tensor:
        """
            note 一次推理
        """

        # NOTE 文本向量化
        #  对文本进行向量化、注意之前文本是 8 层，但是这里只处理1层就行（虽然音频也一块搞了，但是不会使用
        text_embeddings = self.llm.embed_tokens(input_ids[:, 0, :])

        # NOTE 音频向量化
        # input_ids 形状是 [batch_size * 2, 8, max_len]
        #   * audio_mask .unsqueeze(1)：将形状设置成 [batch_size * 2, 1, max_len]，然后和前边的 input_ids 通过 Broadcasting/广播 对齐，然后做逐元素相乘、将非音频部分设置成0
        #   + self.codebook_layer....view(1,-1,1) 形状是 [1,8,1]，和前边的值做逐元素相加
        #       这 8 个元素和音频做逐元素相加是因为 audio code 在不同层含义不一样、这里每层都和 layer*vocab_size相加、获取码本在当前层的值
        audio_ids = input_ids * audio_mask.unsqueeze(
            1
        ) + self.codebook_layer_offsets.view(1, -1, 1)
        audio_embeddings = self.audio_embeddings(audio_ids).sum(dim=1) # tip 对音频进行向量化，并且对应 seq 的向量进行合并

        # NOTE 组装输入 embedding
        embeddings = torch.where(
            # 条件
            audio_mask.unsqueeze(-1),
            audio_embeddings, # true 时选择 [batch_size * 2, all_seq_len, hidden_size]
            text_embeddings # false 时选择 [batch_size * 2, all_seq_len, hidden_size]
        )

        # NOTE
        hidden_states = self.llm(
            inputs_embeds=embeddings, # tip [batch_size * 2, max_len, hidden_size]
            attention_mask=attention_mask, # tip [batch_size * 2, 1, max_len, max_len]，对于一个请求、所有参与推理的位置的mask都是true、pad位置不是
            # tip bi-direction attention：比如 text 后边 target audio 的token、embedding会随着一轮又一轮的推理一直变化，所以 bi-direction attention 计算、
            #      前边的注意力计算结果也会变化，所以
            use_cache=False,
            return_dict=True,
        ).last_hidden_state # NOTE 【重要】对于 transformer layer/ transformer 主干，输入形状和输出形状不变

        # NOTE audio_heads 的 output_dim 是 vocab_size * 8 ，所以是一下子把8个 audio codes 的结果输出出来了
        #       vocab_size 个元素中选择出一个 audio code，共选择出8个
        #
        # logits 形状是 [batch_size * 2, max_len, vocab_size * 8]
        logits = self.audio_heads(hidden_states)

        # tip 最后一个维度拆开：[batch_size * 2, max_len, vocab_size * 8] -> [batch_size * 2, 8, max_len, vocab_size]
        return logits.view(
            # input_ids 形状是 [batch_size * 2, 8, max_len]
            input_ids.shape[0], # batch_size * 2
            input_ids.shape[-1], # max_len
            self.config.num_audio_codebook, # 8
            self.config.audio_vocab_size, # audio_vocab_size
        ).permute(0, 2, 1, 3) # permute 重新排列、交换了 1、2：等价于 1、2 维度的矩阵进行了转置，参考如下注释

    """
# 形状 [1, 2, 3, 2]
x = torch.tensor([
    [  # 请求 0
        [  # 时间帧 0
            [10, 11],  # 码本 0
            [12, 13],  # 码本 1
            [14, 15],  # 码本 2
        ],
        [  # 时间帧 1
            [20, 21],  # 码本 0
            [22, 23],  # 码本 1
            [24, 25],  # 码本 2
        ],
    ]
])

# 形状 [1, 3, 2, 2]
# 注意：如果有来个请求、即 batch_size = 2，则重排列之后的效果是：每个请求内的 1、2 维矩阵各自转置
y = x.permute(0, 2, 1, 3) 之后
[
    [  # 请求 0
        [  # 码本 0
            [10, 11],  # 时间帧 0              // NOTE [10, 11] 看成1个元素
            [20, 21],  # 时间帧 1
        ],
        [  # 码本 1
            [12, 13],  # 时间帧 0
            [22, 23],  # 时间帧 1
        ],
        [  # 码本 2
            [14, 15],  # 时间帧 0
            [24, 25],  # 时间帧 1
        ],
    ]
]
    """
