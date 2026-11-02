# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xiaomi Corp. (authors: Han Zhu). See LICENSE.
"""OmniVoice 双向注意力主干及八码本输出头。"""

import torch
from torch import nn
from transformers import PreTrainedModel, Qwen3Model

from sglang_omni.models.omnivoice.attention import fuse_qkv_attention
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

    def fuse_qkv_projections(self) -> None:
        """权重加载完成后合并各层 QKV；重复调用不会再次分配权重。"""
        for layer in self.llm.layers:
            # tip 遍历模型每层 transformer layer
            # note 直接复用这一层的 attn 对象，仅融合 QKV 并替换 forward。
            fuse_qkv_attention(layer.self_attn)

    def get_input_embeddings(self) -> nn.Embedding:
        # tip self.llm 初始化的时候赋值 Qwen3TTS 模型， embed_tokens 是这个字段的固定属性
        return self.llm.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,  # [batch_size * 2, num_audio_codebook, max_seq_len]
        audio_mask: torch.Tensor,  # [batch_size * 2, max_seq_len]
        attention_mask: torch.Tensor,  # [batch_size * 2, 1, max_seq_len, max_seq_len]
        *,
        target_position_indices: torch.Tensor, # [x] 一维的 tensor，元素值是目标音频对应的位置；另外之所以是一维是因为将 batch_size * 2 和每个batch中的 max_seq_length 打平了
    ) -> torch.Tensor:
        """完整序列经过主干后，仅投影目标音频位置。

        索引指向展平后的 batch_size * 2 * max_seq_len 个位置；
        返回 [num_audio_codebook, total_target_frames * 2, audio_vocab_size]。
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

        # note
        #  获取结果音频位置的 [x, hidden_size] 数据，后边进行 audio_heads(lm_head)计算的时候仅仅计算这些位置
        #  【重要】每个位置的 audio_heads 计算都是相对独立的、不管是不是在同一个请求中
        #       audio_heads 之后获取8层编码信息之后进行 sample 的时候，才会考虑是否是在同一个请求中（比如一个推理中每个 step 要填充的音频数量不一致）
        #  flatten 作用：[batch_size * 2, max_seq_len, hidden_size] → [total_target_frames * 2, hidden_size]
        target_hidden_states = hidden_states.flatten(0, 1).index_select(
            0, # 表示沿着第0维度，获取这个维度要获取的 target_position_indices 值
            target_position_indices
        )
        logits = self.audio_heads(target_hidden_states)

        # note
        #  结果形状 [8, total_target_frames * 2, audio_vocab_size]
        #  三维tensor，比如 shape = (1,1,3) 时
        #   [
        #       [  // 第一维度
        #          [ // 第二维度
        #              1，2，3 // 第三维度
        #           ]
        #       ]
        #   ]
        #  第三个维度是某一层的audio code 在所有可选的 audio vocab 的概率分布，这个是不变的
        return logits.view(
            # x.shape 是 [total_target_frames * 2, hidden_size], 获取[0] 就是 第一个元素 total_target_frames * 2
            #   另外 .shape 属性类型是 torch.Size
            target_hidden_states.shape[0], # tip total_target_frames * 2
            self.config.num_audio_codebook, # 8
            self.config.audio_vocab_size,
        ).permute(1, 0, 2)

    """
# 以下保留通用的四维 permute 教学示例；本方法现在返回三维目标 logits。
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
