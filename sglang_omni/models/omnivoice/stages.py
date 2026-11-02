# SPDX-License-Identifier: Apache-2.0
"""原生 Stage 工厂、微批处理及完整音频交接。"""

import logging
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import CancelledError
from contextlib import nullcontext
from functools import partial

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


def scheduler(
    # 参数是可调用类型
    compute: Callable[
        [list[StagePayload], CancellationCheck], # 输入
        list[StagePayload | Exception] # 输出
    ],
    #tip
    # device = torch.device("cpu")     # CPU
    # device = torch.device("cuda:0")  # 第 0 张 GPU
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
        """在调度线程中执行推理，不继承创建线程的 CUDA 当前设备。
           NOTE 批处理
        """
        with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
            # NOTE compute 推理在这里执行
            return compute(payloads, scheduler_instance.is_aborted)

    def single(payload: StagePayload) -> StagePayload:
        """单请求必须抛出异常，避免把 Exception 当作成功 payload。
           NOTE 单条任务处理，内部调用 run/批处理
        """
        result = run([payload])[0]
        if isinstance(result, Exception):
            raise result
        return result

    scheduler_instance = SimpleScheduler(
        compute_fn = single,
        batch_compute_fn=run,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_wait_when_idle=False,
    )
    return scheduler_instance


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 2,
) -> SimpleScheduler[StagePayload, StagePayload]:
    """读取 Codec 元数据，CPU 预处理不加载任何模型权重。
       NOTE stage 1
    """
    # tip 仅仅用来获取 sample_rate，没有加载模型
    checkpoint = resolve_checkpoint(model_path)
    codec_config = HiggsAudioV2TokenizerConfig.from_pretrained(
        f"{checkpoint}/audio_tokenizer"
    )

    def preprocess(payload: StagePayload) -> StagePayload:
        """保留请求 ID，完成媒体和参数边界转换。"""
        started = time.perf_counter()
        try:
            # NOTE 音频 -> waveform；生成音频相关的配置
            state = build_omnivoice_state(payload, codec_config.sample_rate)
        except ValueError as error:
            raise ValueError(f"OmniVoice request: {error}") from error
        state.engine_time_s = time.perf_counter() - started
        logger.info(
            "request_id=%s preprocessing 完成，耗时=%.3fs",
            payload.request_id,
            state.engine_time_s,
        )
        # tip 将结果 state 保存到 payload.data，下一个 stage 用 payload.data 获取上一个 stage 的结果
        return store_state(payload, state)

    return SimpleScheduler(
        compute_fn = preprocess,
        max_concurrency=max_concurrency
    )


def encode_batch(
    payloads: list[StagePayload],
    is_cancelled: CancellationCheck,
    *,
    codec: AudioCodec,
    codebooks: int,
) -> list[StagePayload | Exception]:
    """逐条编码参考音频，失败不影响同批其他请求。
       NOTE stage 2 用这个方法： waveform -> [8, codec frame]
    """
    results: list[StagePayload | Exception] = []

    for payload in payloads:
        try:
            if is_cancelled(payload.request_id):
                raise CancelledError(payload.request_id)
            started = time.perf_counter()
            # NOTE 读取上一个阶段的结果、waveform
            state = load_state(payload, OmniVoiceState)
            assert state.reference_waveform is not None
            # NOTE waveform -> audio_codes
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
            for chunk_index, chunk_text in enumerate(state.chunk_texts):
                logger.info(
                    f"request_id={payload.request_id} 文本分段 {chunk_index + 1}/{len(state.chunk_texts)}，"
                    f"目标帧数={state.target_frames[chunk_index]}，"
                    f"预计时长={state.target_frames[chunk_index] / codec.model.config.frame_rate:.2f}s，"
                    f"文本={chunk_text!r}"
                )
            if is_cancelled(payload.request_id):
                raise CancelledError(payload.request_id)
            # NOTE 将结果 state 保存到 payload 中
            #   将 payload 保存到 results 中
            #   [重要] stage 之间传递数据的是这个方法的返回结果， store_state 和 load_state 更新 payload 的「下一阶段需要获取的结果」以及方便下一阶段获取数据
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
    """参考编码器与 vocoder 共用同进程、同设备的 Codec。
       NOTE stage 2
    """
    # tip 结果 resolved 是指用什么硬件运行
    resolved = resolve_concrete_device(device, gpu_id)
    checkpoint = resolve_checkpoint(model_path)
    config = OmniVoiceConfig.from_pretrained(checkpoint)

    logger.info(
        f"参考音频编码器初始化：device={device}, gpu_id={gpu_id}, "
        f"resolved={resolved}, checkpoint={checkpoint}, "
        f"num_audio_codebook={config.num_audio_codebook}"
    )

    return scheduler(
        # partial 主要作用是为 encode_batch 预先绑定 codec 和 codebooks 两个参数，返回值是包装的 encode_batch
        #  tip 原始的 encode_batch 协议和 scheduler 第一个参数协议不一样，但是partial包装之后有两个参数已经预先写上了、协议就对上了
        partial(
            encode_batch, # NOTE 第二步关键
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
    tokenizer: PreTrainedTokenizerBase, # tip 文本tokenizer
) -> list[StagePayload | Exception]:
    """按采样设置分组，跨请求组批、同一请求的长文本各段顺序生成。
       note stage 3 使用该方法
    """
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
        # note 对 batch 的每个请求进行遍历
        try:
            if is_cancelled(payload.request_id):
                raise CancelledError(payload.request_id)
            # tip 获取上一阶段的结果
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
            # tip 这个是按照生成规则对请求进行分组
            groups[state.generation.model_dump_json()].append(index)
        except Exception as error:
            logger.exception("request_id=%s tts_engine 输入失败", payload.request_id)
            results[index] = error
    for indices in groups.values():
        # tip indices 是请求参数相同的 states 的下标
        for chunk_index in range(
            max(len(states[index].chunk_texts) for index in indices)
        ):
            # tip 上一步按照 请求参数对完整的 tts 请求进行了分组，分组后的请求 chunk_text 可能不一样
            #   比如：
            #       请求 A：chunk_texts = ["A第一段", "A第二段", "A第三段"]
            #       请求 B：chunk_texts = ["B第一段"]
            #   这里按照最长的一个请求的 chunk_text 的文本片段数量进行遍历
            active: list[int] = []
            items: list[SamplingItem] = []
            for index in indices:
                # 遍历请求参数相同的 states 元素

                # 过滤异常或者取消的请求
                if isinstance(results[index], Exception) or chunk_index >= len(
                    states[index].chunk_texts
                ):
                    continue
                if is_cancelled(payloads[index].request_id):
                    results[index] = CancelledError(payloads[index].request_id)
                    continue

                state = states[index]
                active.append(index)
                # tip items 保存请求
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
                # NOTE 推理核心代码
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
    dtype: str,
    max_batch_size: int,
    max_batch_wait_ms: int,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    """加载原生 OmniVoice 主干，不创建自回归调度器或 KV 缓存。
       NOTE stage 3
    """
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtypes:
        raise ValueError(f"Unsupported OmniVoice dtype: {dtype}")
    resolved = resolve_concrete_device(device, gpu_id)
    checkpoint = resolve_checkpoint(model_path)
    # NOTE 初始化 stage，初始化 stage 的时候加载模型
    model = (
        OmniVoiceModel.from_pretrained(
            checkpoint, dtype=dtypes[dtype], attn_implementation="sdpa"
        )
        .to(resolved)
        .eval()
        .requires_grad_(False)
    )
    # NOTE 拼接 qkv 的投影proj，后边一次性计算出qkv
    model.fuse_qkv_projections()
    # tip: text - tokenizer -> text token_id
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return scheduler(
        partial(
            generate_batch, # note compute_fn = generate_batch
            model=model,  # tip 已经初始化的模型
            tokenizer=tokenizer # tip 已经初始化的 text tokenizer
        ),
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
    """使用共享 Codec 返回完整波形，不开放增量输出。
       NOTE stage 4
    """
    resolved = resolve_concrete_device(device, gpu_id)
    checkpoint = resolve_checkpoint(model_path)
    return scheduler(
        partial(decode_batch, codec=load_codec(checkpoint, str(resolved))),
        resolved,
        max_batch_size,
        max_batch_wait_ms,
    )
