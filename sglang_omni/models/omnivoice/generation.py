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
    target_frames: int
    generator: torch.Generator


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
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造条件输入；音频位置按码本排列，文本在每个码本轴复制。"""
    state = item.state
    assert state.reference_codes is not None
    codebooks = model.config.num_audio_codebook
    style = "<|denoise|>" if state.generation.denoise else ""
    style += f"<|lang_start|>{state.language or 'None'}<|lang_end|>"
    style += "<|instruct_start|>None<|instruct_end|>"
    style_ids = (
        tokenizer(style, return_tensors="pt")
        .input_ids.repeat(codebooks, 1)
        .to(model.device)
    )
    text = f"<|text_start|>{combine_text(item.text, state.reference_text)}<|text_end|>"
    text_ids = tokenize_text(text, tokenizer).repeat(codebooks, 1).to(model.device)
    reference_codes = state.reference_codes.to(model.device)
    total_positions = (
        style_ids.shape[-1]
        + text_ids.shape[-1]
        + reference_codes.shape[-1]
        + item.target_frames
    )
    if total_positions > model.config.llm_config.max_position_embeddings:
        raise ValueError("OmniVoice prompt is longer than the model's context length")
    target = torch.full(
        (codebooks, item.target_frames),
        model.config.audio_mask_id,
        dtype=torch.long,
        device=model.device,
    )
    input_ids = torch.cat((style_ids, text_ids, reference_codes, target), dim=1)
    audio_mask = torch.zeros(input_ids.shape[-1], dtype=torch.bool, device=model.device)
    audio_mask[style_ids.shape[-1] + text_ids.shape[-1] :] = True
    return input_ids, audio_mask


def pack_batch(
    inputs: list[tuple[torch.Tensor, torch.Tensor]],
    target_frames: list[int],
    mask_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """同时组装条件／无条件两组行，并严格屏蔽不同长度的 padding。"""
    batch_size = len(inputs)
    width = max(ids.shape[-1] for ids, _ in inputs)
    device = inputs[0][0].device
    input_ids = torch.full(
        (2 * batch_size, inputs[0][0].shape[0], width),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    audio_mask = torch.zeros((2 * batch_size, width), dtype=torch.bool, device=device)
    attention_mask = torch.zeros(
        (2 * batch_size, 1, width, width), dtype=torch.bool, device=device
    )
    for index, ((ids, mask), frames) in enumerate(zip(inputs, target_frames)):
        length = ids.shape[-1]
        input_ids[index, :, :length] = ids
        audio_mask[index, :length] = mask
        attention_mask[index, :, :length, :length] = True
        input_ids[batch_size + index, :, :frames] = ids[:, -frames:]
        audio_mask[batch_size + index, :frames] = True
        attention_mask[batch_size + index, :, :frames, :frames] = True
        padding = torch.arange(frames, width, device=device)
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
    scores: torch.Tensor, temperature: float, generator: torch.Generator
) -> torch.Tensor:
    """使用请求自己的随机数源，不修改进程级 RNG。"""
    noise = torch.rand(
        scores.shape, dtype=scores.dtype, device=scores.device, generator=generator
    )
    return scores / temperature - torch.log(-torch.log(noise + 1e-10) + 1e-10)


def predict_tokens(
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    options: GenerationOptions,
    mask_id: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在 log-probability 上进行 CFG，返回候选码本及置信度。"""
    log_probs = conditional.float().log_softmax(dim=-1)
    if options.guidance_scale != 0:
        unconditioned = unconditional.float().log_softmax(dim=-1)
        log_probs = (
            log_probs + options.guidance_scale * (log_probs - unconditioned)
        ).log_softmax(dim=-1)
    log_probs[..., mask_id] = -float("inf")
    if options.class_temperature > 0:
        count = math.ceil(log_probs.shape[-1] * 0.1)
        values, indices = log_probs.topk(count, dim=-1)
        filtered = torch.full_like(log_probs, -float("inf")).scatter(
            -1, indices, values
        )
        predicted = gumbel_scores(
            filtered, options.class_temperature, generator
        ).argmax(dim=-1)
    else:
        predicted = log_probs.argmax(dim=-1)
    return predicted, log_probs.max(dim=-1).values


@torch.inference_mode()
def sample_batch(
    items: list[SamplingItem],
    model: OmniVoiceModel,
    tokenizer: PreTrainedTokenizerBase,
    is_cancelled: CancellationCheck,
) -> list[torch.Tensor | Exception]:
    """一次微批完成所有去掩码步骤；取消不会把半成品当成功结果。"""
    if not items:
        return []
    options = items[0].state.generation
    assert all(item.state.generation == options for item in items)
    original_count = len(items)
    results: dict[int, torch.Tensor | Exception] = {}
    valid_indices: list[int] = []
    valid_items: list[SamplingItem] = []
    inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, item in enumerate(items):
        try:
            if is_cancelled(item.request_id):
                raise CancelledError(item.request_id)
            prepared = prepare_inputs(item, model, tokenizer)
            item.state.prompt_tokens += prepared[0].shape[-1] - item.target_frames
            inputs.append(prepared)
            valid_indices.append(index)
            valid_items.append(item)
        except (ValueError, CancelledError) as error:
            results[index] = error
    items = valid_items
    if not items:
        return [results[index] for index in range(original_count)]
    input_ids, audio_mask, attention_mask = pack_batch(
        inputs, [item.target_frames for item in items], model.config.audio_mask_id
    )
    batch_size = len(items)
    codebooks = model.config.num_audio_codebook
    tokens = [
        torch.full(
            (codebooks, item.target_frames),
            model.config.audio_mask_id,
            dtype=torch.long,
            device=model.device,
        )
        for item in items
    ]
    schedules = [
        unmask_schedule(item.target_frames, codebooks, options) for item in items
    ]
    layer_indices = torch.arange(codebooks, device=model.device).view(-1, 1)
    for step in range(options.num_step):
        cancelled = [is_cancelled(item.request_id) for item in items]
        if all(cancelled):
            break
        logits = model(input_ids, audio_mask, attention_mask).float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("OmniVoice model produced non-finite logits")
        for index, item in enumerate(items):
            count = schedules[index][step]
            if cancelled[index] or count == 0:
                continue
            length, frames = inputs[index][0].shape[-1], item.target_frames
            predicted, scores = predict_tokens(
                logits[index, :, length - frames : length],
                logits[batch_size + index, :, :frames],
                options,
                model.config.audio_mask_id,
                item.generator,
            )
            scores = scores - layer_indices * options.layer_penalty_factor
            if options.position_temperature > 0:
                scores = gumbel_scores(
                    scores, options.position_temperature, item.generator
                )
            scores.masked_fill_(
                tokens[index] != model.config.audio_mask_id, -float("inf")
            )
            positions = scores.flatten().topk(count).indices
            tokens[index].view(-1)[positions] = predicted.flatten()[positions]
            input_ids[index, :, length - frames : length] = tokens[index]
            input_ids[batch_size + index, :, :frames] = tokens[index]
        del logits
    for index, item, codes in zip(valid_indices, items, tokens, strict=True):
        results[index] = (
            CancelledError(f"request_id={item.request_id}")
            if is_cancelled(item.request_id)
            else codes
        )
    return [results[index] for index in range(original_count)]
