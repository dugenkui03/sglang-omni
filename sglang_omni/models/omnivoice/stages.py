# SPDX-License-Identifier: Apache-2.0
"""原生 Stage 工厂、微批处理及完整音频交接。"""

import logging
import time
from collections import defaultdict
from concurrent.futures import CancelledError
from contextlib import nullcontext
from functools import partial
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import (
    AutoTokenizer,
    HiggsAudioV2TokenizerConfig,
    PreTrainedTokenizerBase,
)

from sglang_omni.models.omnivoice.audio import AudioCodec, load_codec, postprocess_audio
from sglang_omni.models.omnivoice.generation import (
    CancellationCheck,
    SamplingItem,
    plan_chunks,
    sample_batch,
)
from sglang_omni.models.omnivoice.hf_config import OmniVoiceConfig
from sglang_omni.models.omnivoice.model import OmniVoiceModel
from sglang_omni.models.omnivoice.payload_types import OmniVoiceState
from sglang_omni.models.omnivoice.request_builders import build_omnivoice_state
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import build_usage, load_state, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)


class BatchCompute(Protocol):
    """由调度器传入取消查询，结果按输入请求顺序返回。"""

    def __call__(
        self, payloads: list[StagePayload], is_cancelled: CancellationCheck
    ) -> list[StagePayload | Exception]:
        """执行阶段计算并隔离可归属到单个请求的失败。"""
        ...


def batch_scheduler(
    compute: BatchCompute,
    device: torch.device,
    max_batch_size: int,
    max_batch_wait_ms: int,
) -> SimpleScheduler[StagePayload, StagePayload]:
    """在线程中绑定设备；同设备默认流保证阶段 Tensor 交接的先后关系。"""
    if max_batch_size < 1 or max_batch_wait_ms < 0:
        raise ValueError(
            "OmniVoice requires positive max_batch_size and nonnegative max_batch_wait_ms"
        )

    @torch.inference_mode()
    def run(payloads: list[StagePayload]) -> list[StagePayload | Exception]:
        """在调度线程中执行推理，不继承创建线程的 CUDA 当前设备。"""
        with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
            return compute(payloads, scheduler.is_aborted)

    def single(payload: StagePayload) -> StagePayload:
        """单请求必须抛出异常，避免把 Exception 当作成功 payload。"""
        result = run([payload])[0]
        if isinstance(result, Exception):
            raise result
        return result

    scheduler = SimpleScheduler(
        single,
        batch_compute_fn=run,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_wait_when_idle=False,
    )
    return scheduler


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 2,
) -> SimpleScheduler[StagePayload, StagePayload]:
    """读取 Codec 元数据，CPU 预处理不加载任何模型权重。"""
    checkpoint = resolve_checkpoint(model_path)
    codec_config = HiggsAudioV2TokenizerConfig.from_pretrained(
        f"{checkpoint}/audio_tokenizer"
    )

    def preprocess(payload: StagePayload) -> StagePayload:
        """保留请求 ID，完成媒体和参数边界转换。"""
        started = time.perf_counter()
        try:
            state = build_omnivoice_state(payload, codec_config.sample_rate)
        except ValueError as error:
            raise ValueError(f"OmniVoice request: {error}") from error
        state.engine_time_s = time.perf_counter() - started
        logger.info(
            "request_id=%s preprocessing 完成，耗时=%.3fs",
            payload.request_id,
            state.engine_time_s,
        )
        return store_state(payload, state)

    return SimpleScheduler(preprocess, max_concurrency=max_concurrency)


def encode_batch(
    payloads: list[StagePayload],
    is_cancelled: CancellationCheck,
    *,
    codec: AudioCodec,
    codebooks: int,
) -> list[StagePayload | Exception]:
    """逐条编码参考音频，失败不影响同批其他请求。"""
    results: list[StagePayload | Exception] = []
    for payload in payloads:
        try:
            if is_cancelled(payload.request_id):
                raise CancelledError(payload.request_id)
            started = time.perf_counter()
            state = load_state(payload, OmniVoiceState)
            assert state.reference_waveform is not None
            state.reference_codes = codec.encode(state.reference_waveform)
            if (
                state.reference_codes.ndim != 2
                or state.reference_codes.shape[0] != codebooks
            ):
                raise ValueError(
                    "OmniVoice reference codec shape does not match checkpoint"
                )
            state.reference_waveform = None
            plan_chunks(state, codec.model.config.frame_rate)
            state.engine_time_s += time.perf_counter() - started
            logger.info(
                "request_id=%s audio_encoder 完成，参考码本=%s，分段数=%s",
                payload.request_id,
                tuple(state.reference_codes.shape),
                len(state.chunk_texts),
            )
            if is_cancelled(payload.request_id):
                raise CancelledError(payload.request_id)
            results.append(store_state(payload, state))
        except Exception as error:
            logger.exception("request_id=%s audio_encoder 失败", payload.request_id)
            payload.data = {}
            results.append(error)
    return results


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    """参考编码器与 vocoder 共用同进程、同设备的 Codec。"""
    resolved = resolve_concrete_device(device, gpu_id)
    checkpoint = resolve_checkpoint(model_path)
    config = OmniVoiceConfig.from_pretrained(checkpoint)
    return batch_scheduler(
        partial(
            encode_batch,
            codec=load_codec(checkpoint, str(resolved)),
            codebooks=config.num_audio_codebook,
        ),
        resolved,
        1,
        0,
    )


def generate_batch(
    payloads: list[StagePayload],
    is_cancelled: CancellationCheck,
    *,
    model: OmniVoiceModel,
    tokenizer: PreTrainedTokenizerBase,
) -> list[StagePayload | Exception]:
    """按采样设置分组，跨请求组批、同一请求的长文本各段顺序生成。"""
    results: list[StagePayload | Exception] = list(payloads)
    states: dict[int, OmniVoiceState] = {}
    generators: dict[int, torch.Generator] = {}
    groups: dict[str, list[int]] = defaultdict(list)
    started = time.perf_counter()
    logger.info(
        "request_ids=%s tts_engine 开始批处理，批次大小=%s",
        [payload.request_id for payload in payloads],
        len(payloads),
    )
    for index, payload in enumerate(payloads):
        try:
            if is_cancelled(payload.request_id):
                raise CancelledError(payload.request_id)
            state = load_state(payload, OmniVoiceState)
            assert state.reference_codes is not None and state.seed is not None
            if not state.chunk_texts or len(state.chunk_texts) != len(
                state.target_frames
            ):
                raise ValueError("OmniVoice encoded request is missing its chunk plan")
            state.generated_codes = []
            states[index] = state
            generators[index] = torch.Generator(device=model.device).manual_seed(
                state.seed
            )
            groups[state.generation.model_dump_json()].append(index)
        except Exception as error:
            logger.exception("request_id=%s tts_engine 输入失败", payload.request_id)
            results[index] = error
    for indices in groups.values():
        for chunk_index in range(
            max(len(states[index].chunk_texts) for index in indices)
        ):
            active: list[int] = []
            items: list[SamplingItem] = []
            for index in indices:
                if isinstance(results[index], Exception) or chunk_index >= len(
                    states[index].chunk_texts
                ):
                    continue
                if is_cancelled(payloads[index].request_id):
                    results[index] = CancelledError(payloads[index].request_id)
                    continue
                state = states[index]
                active.append(index)
                items.append(
                    SamplingItem(
                        request_id=payloads[index].request_id,
                        state=state,
                        text=state.chunk_texts[chunk_index],
                        target_frames=state.target_frames[chunk_index],
                        generator=generators[index],
                    )
                )
            if not items:
                continue
            try:
                codes = sample_batch(items, model, tokenizer, is_cancelled)
                for index, generated in zip(active, codes, strict=True):
                    if isinstance(generated, Exception):
                        results[index] = generated
                    else:
                        states[index].generated_codes.append(generated)
            except Exception as error:
                logger.exception(
                    "request_ids=%s tts_engine 批次失败",
                    [item.request_id for item in items],
                )
                for index in active:
                    results[index] = error
    for index, payload in enumerate(payloads):
        if is_cancelled(payload.request_id):
            results[index] = CancelledError(payload.request_id)
        if isinstance(results[index], Exception):
            payload.data = {}
            continue
        state = states[index]
        state.reference_codes = None
        state.completion_tokens = sum(codes.numel() for codes in state.generated_codes)
        state.finish_reason = "stop"
        state.engine_time_s += time.perf_counter() - started
        logger.info(
            "request_id=%s tts_engine 完成，步数=%s，帧数=%s，seed=%s",
            payload.request_id,
            state.generation.num_step,
            state.target_frames,
            state.seed,
        )
        results[index] = store_state(payload, state)
    return results


def create_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "float16",
    max_batch_size: int = 1,
    max_batch_wait_ms: int = 10,
) -> SimpleScheduler[StagePayload, StagePayload]:
    """加载原生 OmniVoice 主干，不创建自回归调度器或 KV 缓存。"""
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtypes:
        raise ValueError(f"Unsupported OmniVoice dtype: {dtype}")
    resolved = resolve_concrete_device(device, gpu_id)
    checkpoint = resolve_checkpoint(model_path)
    model = (
        OmniVoiceModel.from_pretrained(
            checkpoint, dtype=dtypes[dtype], attn_implementation="sdpa"
        )
        .to(resolved)
        .eval()
        .requires_grad_(False)
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return batch_scheduler(
        partial(generate_batch, model=model, tokenizer=tokenizer),
        resolved,
        max_batch_size,
        max_batch_wait_ms,
    )


def decode_batch(
    payloads: list[StagePayload],
    is_cancelled: CancellationCheck,
    *,
    codec: AudioCodec,
) -> list[StagePayload | Exception]:
    """等帧数码本批量解码，再恢复每个请求的分段顺序。"""
    started = time.perf_counter()
    results: list[StagePayload | Exception] = list(payloads)
    states: dict[int, OmniVoiceState] = {}
    waveforms: dict[tuple[int, int], NDArray[np.float32]] = {}
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index, payload in enumerate(payloads):
        try:
            if is_cancelled(payload.request_id):
                raise CancelledError(payload.request_id)
            state = load_state(payload, OmniVoiceState)
            if not state.generated_codes:
                raise ValueError("OmniVoice has no generated audio codes")
            states[index] = state
            for chunk_index, codes in enumerate(state.generated_codes):
                groups[codes.shape[-1]].append((index, chunk_index))
        except Exception as error:
            results[index] = error
    for group in groups.values():
        # 每次最多合并一个微批的请求数量，避免长文本段数意外放大 GPU 显存。
        for offset in range(0, len(group), len(payloads)):
            selected = [
                (index, chunk)
                for index, chunk in group[offset : offset + len(payloads)]
                if not isinstance(results[index], Exception)
                and not is_cancelled(payloads[index].request_id)
            ]
            if not selected:
                continue
            try:
                decoded = codec.decode(
                    torch.stack(
                        [
                            states[index].generated_codes[chunk]
                            for index, chunk in selected
                        ]
                    )
                )
                for (index, chunk), waveform in zip(
                    selected, decoded.float().cpu().numpy(), strict=True
                ):
                    if not np.isfinite(waveform).all():
                        results[index] = RuntimeError(
                            "OmniVoice codec produced non-finite audio"
                        )
                    else:
                        waveforms[index, chunk] = waveform
            except Exception as error:
                logger.exception(
                    "request_ids=%s vocoder 批次失败",
                    [payloads[index].request_id for index, _ in selected],
                )
                for index, _ in selected:
                    results[index] = error
    for index, payload in enumerate(payloads):
        try:
            if is_cancelled(payload.request_id):
                raise CancelledError(payload.request_id)
            if isinstance(results[index], Exception):
                raise results[index]
            state = states[index]
            waveform = postprocess_audio(
                [
                    waveforms[index, chunk]
                    for chunk in range(len(state.generated_codes))
                ],
                state,
            )
            state.engine_time_s += time.perf_counter() - started
            payload.data = {
                **audio_waveform_payload(
                    waveform, sample_rate=state.sample_rate, modality="audio"
                ),
                "usage": build_usage(state),
                "finish_reason": "stop",
            }
            logger.info(
                "request_id=%s vocoder 完成，采样点=%s，总耗时=%.3fs",
                payload.request_id,
                waveform.size,
                state.engine_time_s,
            )
            results[index] = payload
        except Exception as error:
            payload.data = {}
            results[index] = error
    return results


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_batch_size: int = 1,
    max_batch_wait_ms: int = 10,
) -> SimpleScheduler[StagePayload, StagePayload]:
    """使用共享 Codec 返回完整波形，不开放增量输出。"""
    resolved = resolve_concrete_device(device, gpu_id)
    checkpoint = resolve_checkpoint(model_path)
    return batch_scheduler(
        partial(decode_batch, codec=load_codec(checkpoint, str(resolved))),
        resolved,
        max_batch_size,
        max_batch_wait_ms,
    )
