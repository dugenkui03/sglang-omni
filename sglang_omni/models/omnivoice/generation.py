# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xiaomi Corp. (authors: Han Zhu). See LICENSE.
"""按请求隔离随机状态的迭代去掩码与长文本生成。"""

import math
from concurrent.futures import CancelledError
from dataclasses import dataclass
from typing import Protocol

import torch
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.omnivoice.duration import RuleDurationEstimator
from sglang_omni.models.omnivoice.hf_config import GenerationOptions
from sglang_omni.models.omnivoice.model import OmniVoiceModel
from sglang_omni.models.omnivoice.payload_types import OmniVoiceState
from sglang_omni.models.omnivoice.text import (
    chunk_text_punctuation,
    combine_text,
    tokenize_text,
)


class CancellationCheck(Protocol):
    """调度器提供的只读取消查询，不消耗取消标记。"""

    def __call__(self, request_id: str) -> bool:
        """返回请求是否已经取消。"""
        ...


@dataclass(kw_only=True)
class SamplingItem:
    """一次分段生成的输入，随机数生成器跨段连续使用。"""

    request_id: str
    state: OmniVoiceState
    text: str
    target_frames: int # 这个请求生成的目标帧数量
    generator: torch.Generator # 随机数生成器


def plan_chunks(state: OmniVoiceState, frame_rate: float) -> None:
    """依据参考帧数估算长度，并沿用官方标点分段及速度换算。"""
    assert state.reference_codes is not None
    estimator = RuleDurationEstimator()
    reference_frames = state.reference_codes.shape[-1]
    options = state.generation
    estimated_frames = max(
        1,
        int(
            estimator.estimate_duration(
                state.text, state.reference_text, reference_frames
            )
        ),
    )
    speed = state.speed
    if options.duration is not None:
        target_frames = max(1, int(options.duration * frame_rate))
        speed = estimated_frames / target_frames
    else:
        target_frames = max(
            1,
            int(
                estimator.estimate_duration(
                    state.text, state.reference_text, reference_frames
                )
                / speed
            ),
        )
    if target_frames <= int(options.audio_chunk_threshold * frame_rate):
        state.chunk_texts = [state.text]
        state.target_frames = [target_frames]
        return
    characters_per_chunk = int(
        options.audio_chunk_duration * frame_rate / (target_frames / len(state.text))
    )
    state.chunk_texts = chunk_text_punctuation(
        state.text, characters_per_chunk, min_chunk_len=3
    )
    state.target_frames = [
        max(
            1,
            int(
                estimator.estimate_duration(
                    text, state.reference_text, reference_frames
                )
                / speed
            ),
        )
        for text in state.chunk_texts
    ]


def prepare_inputs(
    item: SamplingItem,
    model: OmniVoiceModel,
    tokenizer: PreTrainedTokenizerBase, # tip 文本 tokenizer
) -> tuple[torch.Tensor, torch.Tensor]:
    """文本 token、参考 audio codes、目标 mask 组成输入

       结果两个元素：
            1）[8, stype_len + text_len + ref_audio_len + target_audio_len] 形状的 输入+结果 tensor
            2）audio mask：一维的 [stype_len + text_len + ref_audio_len + target_audio_len]，元素是 bool 值、true 用来标记后边 ref_audio_len + target_audio_len 的音频位置
    """
    state = item.state
    assert state.reference_codes is not None
    # 默认是8
    num_audio_codebook = model.config.num_audio_codebook

    # note 对 style text 进行 token 化
    style = "<|denoise|>" if state.generation.denoise else ""
    style += f"<|lang_start|>{state.language or 'None'}<|lang_end|>"
    # FIXME 这里用于支持 voice design，暂不支持
    style += "<|instruct_start|>None<|instruct_end|>"
    style_ids = (
        # tip token化并重复8次
        tokenizer(style, return_tensors="pt")
        .input_ids.repeat(num_audio_codebook, 1)
        .to(model.device)
    )
    # tip
    text = f"<|text_start|>{combine_text(item.text, state.reference_text)}<|text_end|>"
    text_ids = tokenize_text(text, tokenizer).repeat(num_audio_codebook, 1).to(model.device)
    reference_codes = state.reference_codes.to(model.device)
    # tip 包含已生成和待生成的序列长度，形状是：[8, seq_len]
    total_positions = (
        style_ids.shape[-1]
        + text_ids.shape[-1]
        + reference_codes.shape[-1] # 参考音频帧数
        + item.target_frames # 按照目标时长计算出的
    )
    if total_positions > model.config.llm_config.max_position_embeddings:
        raise ValueError("OmniVoice prompt is longer than the model's context length")
    # tip 存放目标结果的 tensor
    target = torch.full(
        # shape [8, 目标 seq len]
        (num_audio_codebook, item.target_frames),
        model.config.audio_mask_id,
        dtype=torch.long,
        device=model.device,
    )

    # 沿序列维度 dim=1 将几个 8, x/y/z_seq_len 的tensor拼接在一起，并保留原来的值
    input_ids = torch.cat((style_ids, text_ids, reference_codes, target), dim=1)

    # Audio Mask [all_seq_len] 形状的一维 tensor，第二行代码将 音频位置 mask 值设置为 true
    audio_mask = torch.zeros(input_ids.shape[-1], dtype=torch.bool, device=model.device)
    audio_mask[style_ids.shape[-1] + text_ids.shape[-1] :] = True
    return input_ids, audio_mask


def pack_batch(
    #tip
    # inputs：长度为 n 的列表，列表的每个元素对应一个请求的tensor
    # ├─ inputs[0]：Tensor[8, 20]、Tensor[20]
    # └─ inputs[1]：Tensor[8, 30]、Tensor[30]
    inputs: list[tuple[torch.Tensor, torch.Tensor]],
    target_frames: list[int],
    mask_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
       NOTE 获取组批后的模型输入，由三个 tensor 组成：input ids、audio mask、attn mask

       示例：只有一个请求，batch_size=1、index=0、length=max_seq_len=5、frames=2。
           style、text、ref 各简化为一个位置，目标音频有两帧；编号仅用于教学。
           条件分支没有 Padding；无条件分支只有两个目标位置，需要补齐三个位置。

       ① input_ids：输入编号，完整 shape=[2, 8, 5]，dtype=torch.long。

                       条件分支 input_ids[0]            | 无条件分支 input_ids[1]
                       style  text  ref  target0 target1 | target0 target1    pad        pad        pad
               码本 0   [10,   20,    3,   1024,   1024] | [1024,   1024,   1024/init, 1024/init,  1024/init]
               码本 1   [10,   20,    4,   1024,   1024] | [1024,   1024,   1024/init, 1024/init,  1024/init]
               码本 2   [10,   20,    5,   1024,   1024] | [1024,   1024,   1024/init, 1024/init,  1024/init]
               码本 3   [10,   20,    6,   1024,   1024] | [1024,   1024,   1024/init, 1024/init,  1024/init]
               码本 4   [10,   20,    7,   1024,   1024] | [1024,   1024,   1024/init, 1024/init,  1024/init]
               码本 5   [10,   20,    8,   1024,   1024] | [1024,   1024,   1024/init, 1024/init,  1024/init]
               码本 6   [10,   20,    9,   1024,   1024] | [1024,   1024,   1024/init, 1024/init,  1024/init]
               码本 7   [10,   20,   10,   1024,   1024] | [1024,   1024,   1024/init, 1024/init,  1024/init]

       ② audio_mask：音频位置掩码，完整 shape=[2, 5]，dtype=torch.bool
                    条件分支 audio_mask[0]      // tip 多出来的音频是参考音频
               [False, False, True, True, True]
                    无条件分支 audio_mask[1]
               [True, True, False, False, False]

       ③ attention_mask：注意力掩码，完整 shape=[2, 1, 5, 5]，dtype=torch.bool。
           条件分支 attention_mask[0, 0]             | 无条件分支 attention_mask[1, 0]
           行列顺序：style/text/ref/target0/target1   | 行列顺序：target0/target1/pad/pad/pad
               [[1, 1, 1, 1, 1],                     | [[1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],                     |  [1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],                     |  [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],                     |  [0, 0, 0, 1, 0],
                [1, 1, 1, 1, 1]]                     |  [0, 0, 0, 0, 1]]
           条件分支：length=max_seq_len，没有 Padding，整张矩阵为 True。
           无条件分支：两个目标位置可以相互关注；padding=[2, 3, 4] 只开放各自对角线。
    """

    # tip 遍历所有的请求中，输入+输出 tensor 最后一个维度、获取请求中 seq_len 的最大值
    max_seq_len = max(ids.shape[-1] for ids, _ in inputs)
    device = inputs[0][0].device
    batch_size = len(inputs)
    # note input_ids 形状是 [ batch_size * 2, 8, max_seq_len]
    input_ids = torch.full(
        (
            2 * batch_size,
            # 8：inputs[0][0] 获取的是第一个请求的、input_ids，其 .shape[0] 是 8/表示每一帧的 codebook 数量
            inputs[0][0].shape[0],
            # 所有请求中的最大 seq_len
            max_seq_len
        ),
        mask_id, # 1024，音频码本一共 1024个值（8 个码本表示一个音频帧，所以每个音频帧是有 1024^8 个值的组合)
        dtype=torch.long,
        device=device,
    )
    # 标识是否是音频的 mask
    audio_mask = torch.zeros((2 * batch_size, max_seq_len), dtype=torch.bool, device=device)
    # NOTE 最后两个维度组成了[max_seq_len, max_seq_len]的矩阵
    #   其中 (x,y)位置的bool值标识计算 x 位置的 hidden_state 时是否可以关注 y 位置的 token 对应的 kv 信息/注意力计算阶段 (y是文本token 或者 某个位置的8层 audio code 的信息)
    #   默认值是false
    attention_mask = torch.zeros(
        (2 * batch_size, 1, max_seq_len, max_seq_len),
        dtype=torch.bool,
        device=device
    )
    for index, ((ids, mask), frames) in enumerate(zip(inputs, target_frames)):
        # NOTE
        #   inputs是list、逻辑形状是[batch_size, 2,(8,all_seq_len)或者(all_seq_len)]，注意，每个请求的 all_seq_len 不相同
        #       all_seq_len 是 style_len + text_len + ref_audio_len + output_audio_len
        #       target_frames 对应 output_audio_len
        #
        # NOTE
        #  index 标识是第几个请求
        #  ((ids, mask), frames) 解释：zip将两个数组 inputs 和 target_frames 按位置配队，之前两者每个元素是 tuple[torch.Tensor, torch.Tensor] 和 int，
        #                           这里直接打包成嵌套的 tuple[tuple[torch.Tensor, torch.Tensor], int]
        #                           ((ids, mask), frames) 直接将右侧打包的结构 unpacking 成 ((ids, mask), frames)
        #                               ids: input tensor，形状是 [8, all_seq_len]
        #                               mask 是 [all_seq_len] 的一维tensor
        #                               frames 对应 target_seq_len

        # tip all_seq_len
        length = ids.shape[-1]
        # tip input_ids 形状是 [ batch_size * 2, 8, max_seq_len]
        #   这里是把真实的请求放到条件分支矩阵中
        #   先按 max_seq_len 统一分配并填入 mask_id，再用 ids 覆盖前 length 个位置。
        #   未覆盖的 max_seq_len-length 个位置才是 Padding；length==max_seq_len 时没有 Padding。
        #
        # ① 条件分支：input_ids[index]
        #   本例只有一个请求，length=max_seq_len=5，所以条件分支末尾没有 Pad。
        #
        #          style  text  ref  target0  target1
        # 码本 0    [10,   20,    3,   1024,   1024]
        # 码本 1    [10,   20,    4,   1024,   1024]
        # 码本 2    [10,   20,    5,   1024,   1024]
        # 码本 3    [10,   20,    6,   1024,   1024]
        # 码本 4    [10,   20,    7,   1024,   1024]
        # 码本 5    [10,   20,    8,   1024,   1024]
        # 码本 6    [10,   20,    9,   1024,   1024]
        # 码本 7    [10,   20,   10,   1024,   1024]
        input_ids[index, :, :length] = ids
        # tip 给 audio_mask 变量赋值
        audio_mask[index, :length] = mask
        # NOTE
        #   batch中的一个请求占用的空间是 [2,1, max_length, max_length]
        #   分别是 [index, 1, max_length, max_length] 和 [batch_size + index, ....]
        #       第一个矩阵的有效token空间都是true，在这里设置值
        #
        # tip 例子：length=max_seq_len=5，条件分支没有 Padding，整张矩阵为 True。
        #          列 y →  0  1  2  3  4
        #   行 x ↓
        #     0           [1, 1, 1, 1, 1]
        #     1           [1, 1, 1, 1, 1]
        #     2           [1, 1, 1, 1, 1]
        #     3           [1, 1, 1, 1, 1]
        #     4           [1, 1, 1, 1, 1]
        attention_mask[index, :, :length, :length] = True
        # note input_ids 形状是 [ batch_size * 2, 8, max_seq_len]
        #      ids 是 input+target 的 tensor、还没补齐到 max_length
        #   所以这里的赋值是将结果位置的值赋值给 UnConditional Input IDs 的前 real_target_length
        #
        # ② 无条件分支：input_ids[batch_size + index]
        #          target0 target1      pad         pad         pad
        # 码本 0    [1024,  1024,   1024/init,  1024/init,  1024/init]
        # 码本 1    [1024,  1024,   1024/init,  1024/init,  1024/init]
        # 码本 2    [1024,  1024,   1024/init,  1024/init,  1024/init]
        # 码本 3    [1024,  1024,   1024/init,  1024/init,  1024/init]
        # 码本 4    [1024,  1024,   1024/init,  1024/init,  1024/init]
        # 码本 5    [1024,  1024,   1024/init,  1024/init,  1024/init]
        # 码本 6    [1024,  1024,   1024/init,  1024/init,  1024/init]
        # 码本 7    [1024,  1024,   1024/init,  1024/init,  1024/init]
        input_ids[batch_size + index, :, :frames] = ids[:, -frames:]
        audio_mask[batch_size + index, :frames] = True
        # NOTE
        #   注意，这里处理的是 index 的第二个 [1, max_length, max_length] 的值
        #   这里前frame个位置的元素设置成了 true-注意，这里是真实的target长度
        attention_mask[batch_size + index, :, :frames, :frames] = True

        # NOTE
        #   torch.arange(a,b) 是一个从 a 开始，到 b-1 结束的一维 tensor
        #       padding ，注意、开始是 frames、结束是 max_length，这里是 target 区域之外的 padding、补齐位置
        #        frames 到 max_seq_len-1 的长度：
        #                       = style_len + text_len + ref_audio_len + (max_seq_len - length)
        #                       = 条件输入的前缀长度 + 条件分支原本需要的补齐长度
        #   第一行是给一个对角线赋值，操作和下边代码同理
        #         for p in padding:
        #             attention_mask[batch_size + index, :, p, p] = True
        #
        # tip 例子：frame = 2, max_length = 5  // 下边这个对角线是为了让计算进行下去，但是计算结果的时候并不会使用对应位置来进行计算
        #          列 y →  0  1  2  3  4
        #   行 x ↓
        #     0           [1, 1, 0, 0, 0]
        #     1           [1, 1, 0, 0, 0]
        #     2           [0, 0, 1, 0, 0]
        #     3           [0, 0, 0, 1, 0]
        #     4           [0, 0, 0, 0, 1]
        padding = torch.arange(frames, max_seq_len, device=device)
        attention_mask[batch_size + index, :, padding, padding] = True
    return input_ids, audio_mask, attention_mask


def unmask_schedule(
    frames: int, codebooks: int, options: GenerationOptions
) -> list[int]:
    """沿用官方时间变换，末轮填完所有剩余位置。"""
    times = torch.linspace(0, 1, options.num_step + 1)
    times = (options.t_shift * times / (1 + (options.t_shift - 1) * times)).tolist()
    total = remaining = frames * codebooks
    counts: list[int] = []
    for step in range(options.num_step):
        count = (
            remaining
            if step == options.num_step - 1
            else min(math.ceil(total * (times[step + 1] - times[step])), remaining)
        )
        counts.append(count)
        remaining -= count
    return counts


def gumbel_scores(
    scores: torch.Tensor, # 某一个推理的多个位置的对数概率值
    temperature: float, # 温度
    generator: torch.Generator
) -> torch.Tensor:
    """
        这里是发挥 temperature 的作用，修改 scores 中的对数概率值（把对数概率值理解成分数就行、分数不一定是logits）
    """
    # note 【重要】这个 tensor 中的随机数影响了 scores 中每个元素的相对大小（温度值决定了这个随机数对相对大小影响的难易程度）
    noise = torch.rand(
        scores.shape,
        dtype=scores.dtype,
        device=scores.device,
        generator=generator
    )
    # 等价表达
    # gumbel_noise = -torch.log(-torch.log(noise + 1e-10) + 1e-10)
    # return scores / temperature + gumbel_noise
    # note temperature 越大， scores / temperature 的值越小、scores / temperature 相对于 gumbel_noise 越小，score 在最终的结果中影响越小
    #   由此实现温度值影响随机性
    return scores / temperature - torch.log(-torch.log(noise + 1e-10) + 1e-10)


def predict_tokens(
    # NOTE cond 和 uncond 返回的结果，形状都是 [8, target_audio_frames, audio_vocab_size]，audio_vocab_size 中每个元素都是这个元素相对于其他元素被选中的分数
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    options: GenerationOptions,
    mask_id: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在 log-probability 上进行 CFG，返回候选码本及置信度。"""

    # tip dim = -1
    #   对 dim=-1 的元素列表转换成对数概率
    #   计算步骤（假设每个元素分别是 s1, s2, ... sn，n 是 hidden_size）
    #       1）计算每个元素中间结果：e^s1 ....
    #       2) 计算出每个元素的概率：中间结果 e^s1 除以 sum（e^sx)，得到概率值 prob_1, prob_2 ...
    #       3) 计算出对数概率 ln(prob_1) ...
    #   结果形状还是 [8, target_audio_frames, audio_vocab_size]
    log_probs = conditional.float().log_softmax(dim=-1)

    if options.guidance_scale != 0:
        # tip 计算出 uncondition 链路的 log prob
        unconditioned = unconditional.float().log_softmax(dim=-1)
        log_probs = (
            # note
            #   log_probs - unconditioned：计算的是每个 audio_code 位置中、每个元素因为加上条件，其被选中的概率变化
            #   guidance_scale：条件因素放大因子
            #   log_probs + x：加强条件因素对元素选中概率的影响
            log_probs + options.guidance_scale * (log_probs - unconditioned)
        ).log_softmax(dim=-1) # 在求一次对数概率

    # tip
    #   log_probs[..., mask_id] 等价于 log_probs[:, :, mask_id]，其中 ... 补全前边所有的维度， mask_id 标识获取 mask_id 位置的元素
    #   note 整体效果就是将 [8, target_audio_frames, audio_vocab_size] 中最后一维 audio_vocab_size 中下标是 mask_id 的元素设置为负无穷
    log_probs[..., mask_id] = -float("inf")

    if options.class_temperature > 0:
        # audio_vocab_size * 0.1
        count = math.ceil(log_probs.shape[-1] * 0.1)

        # note 在最后一维的 audio_vocab_size 个元素中，选择出最大的 count 个元素、也就是 10% 的元素
        #   value 前10%最大的元素， indices 是这些元素对应的下标、可以映射到 audio_code 上
        #   value 和 indices 形状是 [8, target_audio_frames, audio_vocab_size/10]
        # 作用是 log_probs 中每个位置、仅仅获取到前 10% 的元素，其他地方都设置为 负无穷
        values, indices = log_probs.topk(count, dim=-1)
        filtered = torch.full_like(
            log_probs,  # 创建和 log_probs 相同形状的tensor
            -float("inf") # 所有元素都用负无穷
        ).scatter(
            dim = -1, # 沿着最后一个维度
            index = indices, # 仅仅在指定下标的位置、按照顺序填充 src 的值
            src = values
        )
        # tip 通过维度值修改每个元素被选中的概率
        predicted = gumbel_scores(
            filtered, # tip 每个元素是概率值
            options.class_temperature, # 温度
            generator #
        ).argmax(dim=-1) # NOTE argmax 表示沿着最后一维找最大值、注意，返回的是最大值下标
    else:
        predicted = log_probs.argmax(dim=-1)

    # predicted 形状是 [8, target_audio_frames]，标识每个位置实际选取的 audio code 的位置
    # log_probs.max(dim=-1).values 是不考虑温度值、每个位置选取的 audio_code 对应的分数
    #   note 注意，log_probs 形状是 [8, target_audio_frames, audio_vocab_size]，此处返回的是 audio_vocab_size 中最大的一个值、所以形状是 [8, target_audio_frames]
    return predicted, log_probs.max(dim=-1).values


@torch.inference_mode()
def sample_batch(
    items: list[SamplingItem], # note 每一个请求都是一个元素
    model: OmniVoiceModel,
    tokenizer: PreTrainedTokenizerBase, # tip 文本tokenizer
    is_cancelled: CancellationCheck,
) -> list[torch.Tensor | Exception]:
    """
       1. 组成输入： 文本 token、参考 audio codes、目标 mask
       2. 推理
       3. 采样
    """
    if not items:
        return []
    options = items[0].state.generation
    assert all(item.state.generation == options for item in items)
    original_count = len(items)
    results: dict[int, torch.Tensor | Exception] = {}
    valid_indices: list[int] = []
    valid_items: list[SamplingItem] = []
    # note: inputs 两个 tensor 的内容
    #   [8, sequence_length]：style + 文本 token + 参考音频 codes + 目标音频 mask
    #   [sequence_length]: mask，标记哪些地方是音频
    inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, item in enumerate(items):
        try:
            if is_cancelled(item.request_id):
                raise CancelledError(item.request_id)
            # note step 1 合并输入、计算输出长度，来初始化最初的推理输入tensor
            #   prepared[0] 是 输入+输出 tensor
            #   prepared[1]是 audio mask
            prepared = prepare_inputs(item, model, tokenizer)
            # prompt_tokens 是指输入文本+音频
            item.state.prompt_tokens += prepared[0].shape[-1] - item.target_frames
            inputs.append(prepared)
            valid_indices.append(index)
            valid_items.append(item)
        except (ValueError, CancelledError) as error:
            results[index] = error
    items = valid_items
    if not items:
        return [results[index] for index in range(original_count)]

    # note step 1.1
    #   获取 [batch_size * 2, ...] 形状的 input ids、audio mask、atten mask，具体数据分布见方法注释
    input_ids, audio_mask, attention_mask = pack_batch(
        # inputs：长度为 n 的列表，列表的每个元素对应一个请求的tensor
        # ├─ inputs[0]：Tensor[8, 20]、Tensor[20]
        # └─ inputs[1]：Tensor[8, 30]、Tensor[30]
        inputs,
        [item.target_frames for item in items],
        model.config.audio_mask_id
    )

    batch_size = len(items)
    max_sequence_length = input_ids.shape[-1] # tip 就是 input 的 seq length

    # note 条件分支中目标音频各帧的位置下标
    conditional_target_indices: list[torch.Tensor] = []
    # note 同上，但是是 非条件分支
    unconditional_target_indices: list[torch.Tensor] = []
    # note 这里保存的不是原始输入 tensor 中的信息
    #   而是 forward 结束之后，会仅仅获取 target audio 位置的 hidden states 序列，形状是 [target_audio_frames * 2, hidden_size]
    #   这个列表每个元素保存的是获取从第一个请求开始的对应结果在 [target_audio_frames * 2, hidden_size] 中的下标
    target_frame_offsets: list[int] = []
    total_target_frames = 0

    # 每段生成只构造一次索引，所有轮次复用；已填好的目标位置仍保留。
    for request_index, ((request_input_ids, _), item) in enumerate(
        zip(inputs, items, strict=True)
    ):
        """
            request_index 数据下标，从0开始
            request_input_ids：包括 text, ref_audio，注意这个是原始输入
            item: SamplingItem
        """

        # note 保存「条件分支中目标音频各帧的位置下标」。一维 tensor、长度同目标音频frame数量
        target_positions = torch.arange(
            item.target_frames, # 目标音频长度，每一帧40ms
            dtype=torch.long,
            device=model.device
        )

        # tip 条件分支中目标音频各帧的位置下标
        conditional_target_indices.append(
            # 加上一维 Tensor，得到所有目标帧的具体位置
            # tip 注意，target_positions 是一维tensor，后边都是整数，所以后边的整数都是对这个tensor元素的操作
            target_positions
            + request_index * max_sequence_length # 整数，该请求展平后的起始位置：0、max_sequence_length、2*max_sequence_length……
            + request_input_ids.shape[-1] # 整数，该请求补齐前的实际长度，包含上下文和目标音频位置
            - item.target_frames # 目标帧数：减去目标帧数得到目标音频区域的起始位置
        )

        # tip： 同上，但是保存的是 unconditional 的结果位置下标
        unconditional_target_indices.append(
            target_positions
            + (batch_size + request_index) * max_sequence_length
        )
        # tip
        #   第一个元素是0
        #   后边的每个都是目标音频长度
        target_frame_offsets.append(total_target_frames)
        # 总的目标音频长度
        total_target_frames += item.target_frames

    # cat 表示按照 dim 维度进行拼接
    #   指定维度的长度相加，元素数值不变
    #   这里每个 Tensor 都是一维的，拼接后仍是一维，不会增加 batch 维度
    # 拼接结果 shape=[2 * total_target_frames]
    #   total_target_frames 是所有请求的目标音频帧数之和
    #   元素是目标位置在展平后的 hidden_states 中的全局下标
    #   全部条件分支的目标下标在前，全部无条件分支的目标下标在后
    target_position_indices = torch.cat(
        # 列表相加就是拼接成一个列表
        conditional_target_indices + unconditional_target_indices,
        dim=0 # 默认值
    )

    # 默认值是8
    num_audio_codebook = model.config.num_audio_codebook
    # tokens 是个list，每个元素保存的是每个请求中 [8, target_audio_frames] 个位置选中的 audio code 编号
    #   随着每一步推理，里边的元素逐渐被填充
    #   用来辅助判断某个位置是否已经获取了结果
    tokens = [
        torch.full(
            # 形状 [8, target_audio_frames]
            (num_audio_codebook, item.target_frames),
            # 元素默认值
            model.config.audio_mask_id, # 1024
            dtype=torch.long,
            device=model.device,
        )
        for item in items
    ]
    # 记录每轮填写多少 token
    schedules = [
        unmask_schedule(item.target_frames, num_audio_codebook, options) for item in items
    ]
    layer_indices = torch.arange(num_audio_codebook, device=model.device).view(-1, 1)

    # NOTE step 2
    #   进行 num_step 轮推理，逐步填充结果位置数据
    for step in range(options.num_step):
        cancelled = [is_cancelled(item.request_id) for item in items]
        if all(cancelled):
            break

        # NOTE 注意，这里仅仅返回的是 目标音频区域的 logits 信息，形状是 [num_audio_codebook, total_target_frames * 2, audio_vocab_size]
        logits = model(
            input_ids,
            audio_mask,
            attention_mask,
            target_position_indices=target_position_indices,
        ).float()

        if not torch.isfinite(logits).all():
            raise RuntimeError("OmniVoice model produced non-finite logits")
        for index, item in enumerate(items):
            """遍历batch 每一个输入"""

            # 当前轮次填充的 audio code 数量
            count = schedules[index][step]
            if cancelled[index] or count == 0:
                continue

            # 总的有效长度：包括输入和输出
            # note 和结果长度：仅仅包括输出
            length, frames = inputs[index][0].shape[-1], item.target_frames

            # tip 第 x 个请求在对应的结果在 logits 中的开始位置如下
            target_frame_offset = target_frame_offsets[index]
            unconditional_frame_offset = total_target_frames + target_frame_offset # 补一个 total_target_frames 就好

            # NOTE 按照 CFG(classifier free guidance) 和温度值选择 8 * target_frames 位置的 audio code
            #   注意，predicted 是不管有没有温度值、每个位置最终选择的 audio code 的位置
            #       scores 是不考虑温度值，每个位置最大对数概率值
            #       原因是：选择填充的位置 score、选择置信度最大的位置，但是填充的时候允许温度影响、选择到另外一个值
            #   形状都是 [8, target_frames]
            predicted, scores = predict_tokens(
                # tip 第一个维度是8，然后从开始位置、截取目标音频长度的位置
                logits[:, target_frame_offset : target_frame_offset + frames],
                # 同上，unconditional
                logits[
                    :, unconditional_frame_offset : unconditional_frame_offset + frames
                ],
                options,
                model.config.audio_mask_id,
                item.generator,
            )

            # note 这个是对 scores 中的数据按元素做减法
            #   层数越大，惩罚力度越大
            #   scores 中的数据实际是对数概率
            scores = scores - layer_indices * options.layer_penalty_factor

            if options.position_temperature > 0:
                # tip 这个温度值影响的是，在 [8, target_audio_frames] 中选择哪些位置填充结果audio code的时候加上随机因素
                scores = gumbel_scores(
                    scores,
                    options.position_temperature,
                    item.generator
                )

            # 已经选中的元素，即结果不是audio_mask_id的元素，直接设置为负无穷
            scores.masked_fill_(
                # note
                #   tokens 中保存了已经选中的元素，tokens[index] 中保存了第 n 个请求已经被选中的元素，形状是 [8, target_audio_frames]
                #   tokens[index] != model.config.audio_mask_id 是tensor的逐元素计算：结果形状同原始 tensor tokens[index]，结果元素是每个元素和 audio_mask_id 做 != 比较
                #   tokens[index] 形状不一定要和scores相同，但是要能广播成相同形状
                tokens[index] != model.config.audio_mask_id,
                -float("inf") # the value to fill in with
            )
            # tip scores 中选出分数最高的 count 个元素的位置:flatten 先将 [8, target_audio_frames] 打平成 [8 * target_audio_frames]，然后在选择 top count 个最大的值的位置
            positions = scores.flatten().topk(count).indices
            #   positions 形状是 [count]，保存了选择 [8 * target_audio_frames] 中哪些位置的元素
            #   predicted.flatten() [8 * target_audio_frames]，保存了每个位置最终选择的元素的位置，比如[5, ...] 标识第一个位置选择了第5个 audio code
            #   tip predicted.flatten()[positions] 标识：选中了 positions 这些位置、返回这些位子选中的audio code 次序（返回选中位置的选中元素），然后赋值给 tokens[index].view(-1)[positions]
            tokens[index].view(-1)[positions] = predicted.flatten()[positions]

            # NOTE 【很重要】将最终选择的 audio code 放到原始输入：conditional 和 unconditional 中
            #   index：第几个请求
            #   :, ：第几层 audio codebook
            #   length-frames: length：目标音频位置
            input_ids[index, :, length - frames : length] = tokens[index] # tip 选中的 audio code 在 audio vocab 中的位置就是 audio code 的值
            input_ids[batch_size + index, :, :frames] = tokens[index]
        del logits
    for index, item, codes in zip(valid_indices, items, tokens, strict=True):
        results[index] = (
            CancelledError(f"request_id={item.request_id}")
            if is_cancelled(item.request_id)
            else codes
        )
    return [results[index] for index in range(original_count)]
