# SPDX-License-Identifier: Apache-2.0
"""Device graphs for the Qwen3-TTS speaker encoder at bucketed mel lengths."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import smallest_bucket
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.utils.device import device_guard, supports_device_streams
from sglang_omni.utils.logging import json_for_log

logger = logging.getLogger(__name__)

SPEAKER_MEL_N_FFT = 1024
SPEAKER_MEL_HOP = 256
SPEAKER_MEL_FMIN = 0
SPEAKER_MEL_FMAX = 12000
SPEAKER_MEL_CLIP = 1e-5


def reflect_index(length: torch.Tensor, width: int, pad: int) -> torch.Tensor:
    """Gather index of a reflect pad of pad frames around the first length frames
    of a width frame buffer."""
    position = torch.arange(-pad, width + pad, device=length.device)
    index = position.abs()
    over = index - (length - 1)
    index = torch.where(over > 0, (length - 1) - over, index)
    return index.clamp(0, width - 1)


def time_delay_block(
    block: torch.nn.Module, x: torch.Tensor, indices: dict[int, torch.Tensor]
) -> torch.Tensor:
    conv = block.conv
    pad = conv.dilation[0] * (conv.kernel_size[0] - 1) // 2
    if pad:
        x = x.index_select(2, indices[pad])
    else:
        pass
    return F.relu(F.conv1d(x, conv.weight, conv.bias, dilation=conv.dilation))


def masked_mean(
    x: torch.Tensor, mask: torch.Tensor, length: torch.Tensor
) -> torch.Tensor:
    """Mean of x (1, channels, width) over its first length frames, accumulated and
    divided in the precision torch.mean uses and rounded to the dtype of x once."""
    accumulate = torch.promote_types(x.dtype, torch.float32)
    return ((x * mask).sum(2, keepdim=True, dtype=accumulate) / length).to(x.dtype)


def encode_bucketed(
    encoder: torch.nn.Module,
    mels: torch.Tensor,
    length: torch.Tensor,
    pads: Iterable[int],
) -> torch.Tensor:
    """The encoder's forward over its own weights, exact for the first length frames
    of mels (1, mels, width) whatever fills the rest of the width."""
    width = mels.shape[2]
    indices = {pad: reflect_index(length, width, pad) for pad in pads}
    mask = (torch.arange(width, device=mels.device) < length).to(mels.dtype)[None, None]
    x = time_delay_block(encoder.blocks[0], mels, indices)
    outputs = []
    for layer in encoder.blocks[1:]:
        residual = x
        x = time_delay_block(layer.tdnn1, x, indices)
        first, *rest = torch.chunk(x, layer.res2net_block.scale, dim=1)
        parts = [first]
        for block, part in zip(layer.res2net_block.blocks, rest):
            if len(parts) > 1:
                part = part + parts[-1]
            else:
                pass
            parts.append(time_delay_block(block, part, indices))
        x = time_delay_block(layer.tdnn2, torch.cat(parts, dim=1), indices)
        se = layer.se_block
        mean = masked_mean(x, mask, length)
        x = x * torch.sigmoid(se.conv2(F.relu(se.conv1(mean)))) + residual
        outputs.append(x)
    x = time_delay_block(encoder.mfa, torch.cat(outputs, dim=1), indices)
    asp = encoder.asp
    statistics = (
        asp._compute_statistics
    )  # noqa: leading-underscore  # upstream spelling
    mean, std = statistics(x, mask / length.to(x.dtype))
    attention = torch.cat(
        [
            x,
            mean.unsqueeze(2).expand(-1, -1, width),
            std.unsqueeze(2).expand(-1, -1, width),
        ],
        dim=1,
    )
    attention = asp.conv(torch.tanh(time_delay_block(asp.tdnn, attention, indices)))
    attention = F.softmax(attention.masked_fill(mask == 0, float("-inf")), dim=2)
    mean, std = statistics(x, attention)
    return encoder.fc(torch.cat((mean, std), dim=1).unsqueeze(2)).squeeze(-1)[0]


@dataclass
class CapturedSpeakerEncoderGraph:
    graph: ReplayableGraph
    static_mels: torch.Tensor
    static_frames: torch.Tensor
    static_embedding: torch.Tensor


class Qwen3TTSSpeakerEncoderCudaGraphRunner:
    """The speaker embedding of a waveform: the checkpoint's mel front end on the CPU,
    then one captured encode per bucket length, eager past the largest bucket."""

    def __init__(self, encoder: torch.nn.Module, *, sample_rate: int) -> None:
        # NOTE encode 类型参考 Qwen3TTSTalker.__init__()，是 Qwen3TTSSpeakerEncoder（外部引用类）
        self.encoder = encoder
        self.mel_basis = torch.from_numpy(
            librosa_mel_fn(
                sr=sample_rate,
                n_fft=SPEAKER_MEL_N_FFT,
                n_mels=encoder.blocks[0].conv.in_channels,
                fmin=SPEAKER_MEL_FMIN,
                fmax=SPEAKER_MEL_FMAX,
            )
        ).float()
        self.mel_window = torch.hann_window(
            SPEAKER_MEL_N_FFT, dtype=torch.float32, device="cpu"
        )
        self.pads = frozenset(
            conv.dilation[0] * (conv.kernel_size[0] - 1) // 2
            for conv in encoder.modules()
            if isinstance(conv, torch.nn.Conv1d)
        ) - {0}
        self.lock = threading.Lock()
        self.graphs: dict[int, CapturedSpeakerEncoderGraph] = {}
        self.replays = 0
        self.misses = 0

    def mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Log mel (1, mels, frames) of a CPU waveform (1, samples).
        
           将 waveform 转换成 对数梅尔频谱(Log-Mel Spectrogram)，这里没有调用模型/权重，这是个特定算法、其他模型 waveform -> log-mel 也是这些代码

           结果形状参考
            [1, 128, time_frames]
            │   │    └─ 时间帧：time_frames 个分析窗口
            │   └────── 梅尔频带(Mel Frequency Band)：每个窗口有 128 个频带数值  NOTE 这里可以和 声音编码做个类比，声音编码是一帧用 16 个 codebook 编号表示，这里使用 128维的 mel frequency brand 表示
            └────────── 批次：1 段音频
        """
        padding = (SPEAKER_MEL_N_FFT - SPEAKER_MEL_HOP) // 2
        spectrum = torch.stft(
            F.pad(waveform, (padding, padding), mode="reflect"),
            SPEAKER_MEL_N_FFT,
            hop_length=SPEAKER_MEL_HOP,
            win_length=SPEAKER_MEL_N_FFT,
            window=self.mel_window,
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = torch.sqrt(torch.view_as_real(spectrum).pow(2).sum(-1) + 1e-9)
        result = torch.log(
            torch.clamp(torch.matmul(self.mel_basis, magnitude), min=SPEAKER_MEL_CLIP)
        )
        logger.info(
            f"对数梅尔频谱计算完成\n{json_for_log(result)}\n"
            f"第一个时间帧的前 5 个 Mel 值（帧索引 0）：{result[0, :5, 0].tolist()}"
        )
        return result

    def capture(self, codec_frame_buckets: Iterable[int], codec_hop: int) -> None:
        """One graph per bucket of the reference encoder's ladder, in mel frames;
        none when a capture fails, so every clip runs eager."""
        param = next(self.encoder.parameters())
        graph_backend = current_platform.get_device_graph_backend(param.device)
        if not supports_device_streams(param.device) or graph_backend is None:
            return
        else:
            pass
        device_module = torch.get_device_module(param.device)
        buckets = sorted(
            frames * codec_hop // SPEAKER_MEL_HOP for frames in codec_frame_buckets
        )
        if not buckets:
            return
        else:
            pass
        graphs: dict[int, CapturedSpeakerEncoderGraph] = {}
        try:
            with device_guard(param.device):
                pool = graph_backend.graph_pool_handle()
                stream = device_module.Stream(device=param.device)
                # note(ratish): largest first so the shared pool is sized once.
                for frames in reversed(buckets):
                    static_mels = torch.zeros(
                        (1, self.mel_basis.shape[0], frames),
                        device=param.device,
                        dtype=param.dtype,
                    )
                    static_frames = torch.full(
                        (1,), frames, device=param.device, dtype=torch.long
                    )
                    stream.wait_stream(device_module.current_stream(param.device))
                    with torch.inference_mode(), device_module.stream(stream):
                        for _ in range(2):
                            encode_bucketed(
                                self.encoder, static_mels, static_frames, self.pads
                            )
                    with (
                        torch.inference_mode(),
                        graph_backend.capture(
                            pool=pool,
                            stream=stream,
                            thread_local_errors=True,
                        ) as graph,
                    ):
                        static_embedding = encode_bucketed(
                            self.encoder, static_mels, static_frames, self.pads
                        )
                    stream.synchronize()
                    graphs[frames] = CapturedSpeakerEncoderGraph(
                        graph=graph,
                        static_mels=static_mels,
                        static_frames=static_frames,
                        static_embedding=static_embedding,
                    )
        except Exception as exc:
            logger.warning(
                "Qwen3-TTS speaker encoder graph capture disabled the runner: "
                f"{type(exc).__name__}: {exc}",
                exc_info=True,
            )
            return
        self.graphs = graphs
        logger.info(f"Qwen3-TTS speaker encoder graphs captured for {buckets} frames")

    def embed(self, waveform: np.ndarray) -> torch.Tensor:
        """Speaker embedding (enc_dim,) of a waveform at the encoder's sample rate.
        
           NOTE 重要，获取 waveform 对应的 speaker embedding，分为2步：
            1. mel()：波形 → 对数梅尔频谱
            2. speaker_embedding：[2048]
        """

        # 将 waveform 转换成 log-mel 编码
        # tip
        #    结果形状参考
        #     [1, 128, time_frames]
        #     │   │    └─ 时间帧：time_frames 个分析窗口
        #     │   └────── 梅尔频带(Mel Frequency Band)：每个窗口有 128 个频带数值  NOTE 这里可以和 声音编码做个类比，声音编码是一帧用 16 个 codebook 编号表示，这里使用 128维的 mel frequency brand 表示
        #     └────────── 批次：1 段音频
        mels = self.mel(torch.from_numpy(waveform).unsqueeze(0))
        # NOTE 帧数量是按照 帧步长约 10.67ms（256÷24000） 来计算的（这里的帧不是声音编码的 tokenizer 的帧
        frames = mels.shape[2]
        assert frames > max(self.pads)
        param = next(self.encoder.parameters())
        mels = mels.to(param.device).to(param.dtype)
        bucket = smallest_bucket(frames, self.graphs)
        if bucket is None:
            self.misses += 1
            # NOTE 普通链路执行
            speaker_embedding = self.encoder(mels.transpose(1, 2))[0]
            logger.info("说话人编码完成，Mel 与音色向量形状对比\n%s",
                json_for_log({"mel_shape": list(mels.shape), "speaker_embedding_shape": list(speaker_embedding.shape)}))
            return speaker_embedding
        else:
            pass
        captured = self.graphs[bucket]
        # note(ratish): every worker enqueues on one shared stream, so the lock only
        # orders the enqueue and the clone lands before the next replay.
        with self.lock:
            captured.static_mels[:, :, :frames].copy_(mels)
            captured.static_frames.fill_(frames)
            # tip CUDA Graph 执行链路
            captured.graph.replay()
            self.replays += 1
            return captured.static_embedding.clone()
