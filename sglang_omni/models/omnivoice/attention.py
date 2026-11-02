# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team.
"""加载后就地融合 QKV，仅用于推理，不转换检查点保存格式。"""

from types import MethodType
from typing import Unpack

import torch
import torch.nn.functional as F
from torch import nn
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
        # 如果已经是 fused 推理则直接返回
        return

    # tip 将三个 nn.Linear 放入到一个 tuple 中
    projections = (attention.q_proj, attention.k_proj, attention.v_proj)
    # tip 获取这三个 linear 的输出维度
    #     nn.Linear 中有这两个属性，分别表示这个线性层的输入维度和输出维度
    #       self.in_features = in_features
    #       self.out_features = out_features
    attention.qkv_sizes = tuple(layer.out_features for layer in projections)

    # NOTE W_Q、W_K、W_V dim=0 拼接，注意输入向量和 qkv_proj 进行tensor 计算的时候、是与其转置计算
    #   nn.Parameter 类型对象赋值给 nn.Module 之后，可以通过 .parameters() 找到该对象、这是 nn.Parameter 的核心特性
    attention.qkv_weight = nn.Parameter(
        # tip
        #  detach() 是指获取一个和 layer.weight 一样但是不需要梯度计算的 tensor
        #  cat() 中 dim 默认值是0，这个函数将 三个线性层按照 dim=0拼接
        torch.cat([layer.weight.detach() for layer in projections]),
        requires_grad=False # 不为这个权重计算、累积梯度
    )
    attention.qkv_bias = None
    if attention.q_proj.bias is not None:
        # tip bias 也是 attention 的参数，这里拼接三个 bias
        attention.qkv_bias = nn.Parameter(
            torch.cat([layer.bias.detach() for layer in projections]), requires_grad=False
        )
    # 释放原来的三组投影，避免重复占用 GPU 显存；其他配置和层直接复用 tip 牛逼
    del attention.q_proj, attention.k_proj, attention.v_proj
    # note 这是动态绑定方法 dynamic method binding
    #   将 attention.forward 方法和 fused_attention_forward 方法绑定并将 attention 作为 fused_attention_forward 的 self 参数
    attention.forward = MethodType(
        fused_attention_forward,
        attention # note 这个方法说来说去就是更新了 attention 的 qkv_sizes、qkv_weight 和 qkv_bias
    )


def fused_attention_forward(
    self: Qwen3Attention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """W_Q、W_K、W_V 合并后一次计算，再拆分 Q/K/V；后续沿用原注意力逻辑。"""

    # note F.linear(...) 其实是执行了一次线性计算：qkv = hidden_states @ self.qkv_weight.T + self.qkv_bias
    qkv = F.linear(hidden_states, self.qkv_weight, self.qkv_bias)
    # 拆分 qkv
    query_states, key_states, value_states = qkv.split(self.qkv_sizes, dim=-1)

    # hidden_states.shape[:-1] 是指去掉最后一个维度的形状 tuple
    # 比如 hidden_states.shape 是 [batch_size, token_sequence, x] 这里获取的是 [batch_size, token_sequence]
    input_shape = hidden_states.shape[:-1]

    # *input_shape 会把这个 tuple 自动展开为 batch_size, token_sequence
    hidden_shape = (*input_shape, -1, self.head_dim)

    # query_states 原本的形状是 [batch_size, token_sequence, head_num × head_dim]
    #   这里通过 [batch_size, token_sequence, -1  ,head_dim] 进行 reshape 转成 [batch_size, token_sequence, head_num , head_dim]
    #   .transpose(1, 2) 标识交换 1、2维度，得到 [batch_size, head_num, token_sequence, head_dim]
    #
    # tip norm 是均方根归一化，思路是先衡量这组数据有多大，然后用这个大小对所有数据做统一的缩放，避免后续计算过程中出现元素大小膨胀
    #       均方根思路是：先平方、避免正负相互抵消 -> 然后求平均、衡量一个数的典型大小 -> 然后开方、恢复到原来的量纲/数量级
    #           而后，每个数字与上一步获取的值相除
    query_states = self.q_norm(query_states.reshape(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(key_states.reshape(hidden_shape)).transpose(1, 2) # 同上
    # 没有归一化
    # q\v归一化方便控制注意力计算的数据尺度
    # value不参与注意力计算，保留学习到的数据尺度
    value_states = value_states.reshape(hidden_shape).transpose(1, 2)

    # todo 把位置信息融入 Q、K
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        *position_embeddings
    )

    # 选择的是注意力计算方式
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        # eager：把矩阵乘法、Softmax、乘 value 分成多个操作执行
        # FlashAttention： 分块并融合计算，减少中间结果的 GPU 显存读写
        # sdpa：PyTorch 提供的统一接口，根据条件选择具体实现，可能采用 FlashAttention
        self.config._attn_implementation,
        eager_attention_forward
    )

    # NOTE 进行注意力计算
    #   attn_output 是加权后的 value，形状是 [请求数量 × 2, max_sequence_length, head_num, head_dim]         // 注意，这里已经做了补全，而且加上了 unconditional tensor，上边一些描述有问题
    #   attn_weights 是注意力矩阵，比如位置 (x,y) 标识 token_x 对 token_y 的注意力大小、即 token_x_q 对 token_y_k 的向量点积
    #       形状是 [请求数量 × 2, head_num, max_sequence_length, max_sequence_length]
    attn_output, attn_weights = attention_interface(
        self,
        query_states, #
        key_states, #
        value_states, #
        attention_mask,
        dropout=0.0
            if not self.training
            else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )
    # 注意力计算结果转换形状
    # [batch_size × 2, max_sequence_length, head_num, head_dim] -> [batch_size, max_sequence_length, head_num * head_dim]
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    # 对注意力转换接过进行投影
    return self.o_proj(attn_output), attn_weights
