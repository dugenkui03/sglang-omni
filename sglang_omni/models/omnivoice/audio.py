# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Xiaomi Corp. (authors: Han Zhu). See LICENSE.
"""参考音频编解码及官方波形处理。"""

from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from numpy.typing import NDArray
from pydub import AudioSegment
from pydub.silence import detect_leading_silence, split_on_silence
from transformers import HiggsAudioV2TokenizerModel

from sglang_omni.models.omnivoice.payload_types import OmniVoiceState


class AudioCodec:
    """参考编码与输出解码共享权重，并串行访问 Codec 的内部状态。"""

    def __init__(self, checkpoint: str, device: str) -> None:
        """只加载完整检查点自带的 Codec，不隐式下载另一版本。"""
        codec_path = Path(checkpoint) / "audio_tokenizer"
        if not (codec_path / "config.json").is_file():
            raise FileNotFoundError(f"OmniVoice checkpoint requires {codec_path}")
        self.model = (
            HiggsAudioV2TokenizerModel.from_pretrained(
                str(codec_path), dtype=torch.float32
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )
        self.lock = Lock()

    @torch.inference_mode()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """按 Codec 帧步长取完整采样点，不为短输入补造数据。"""
        hop_length = self.model.config.hop_length
        remainder = waveform.shape[-1] % hop_length
        if remainder:
            waveform = waveform[..., :-remainder]
        with self.lock:
            return self.model.encode(
                waveform.unsqueeze(0).to(self.model.device)
            ).audio_codes[0]

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """解码等长度码本批次，保持 [B,1,采样点] 结构。"""
        with self.lock:
            return self.model.decode(codes.to(self.model.device)).audio_values


@lru_cache(maxsize=8)
def load_codec(checkpoint: str, device: str) -> AudioCodec:
    """同一进程、检查点和设备仅保存一份 Codec。"""
    return AudioCodec(checkpoint, device)


def postprocess_audio(
    chunks: list[NDArray[np.float32]], state: OmniVoiceState
) -> NDArray[np.float32]:
    """拼接各段并沿用官方静音、响度和边缘处理顺序。"""
    waveform = cross_fade_chunks(chunks, state.sample_rate)
    if state.vocoder.postprocess_output:
        waveform = remove_silence(
            waveform, state.sample_rate, mid_sil=500, lead_sil=100, trail_sil=100
        )
    if state.reference_rms < 0.1:
        waveform = waveform * state.reference_rms / 0.1
    waveform = fade_and_pad_audio(
        waveform,
        state.vocoder.pad_duration,
        state.vocoder.fade_duration,
        state.sample_rate,
    )
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise RuntimeError("OmniVoice generated empty or non-finite audio")
    return waveform.squeeze(0)


def numpy_to_audiosegment(audio: NDArray[np.float32], sample_rate: int) -> AudioSegment:
    """Convert a numpy float32 array of shape (C, T) to a pydub AudioSegment."""
    audio_int = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    if audio_int.shape[0] > 1:
        audio_int = audio_int.T.flatten()  # interleave channels
    return AudioSegment(
        data=audio_int.tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=audio.shape[0],
    )


def audiosegment_to_numpy(aseg: AudioSegment) -> NDArray[np.float32]:
    """Convert a pydub AudioSegment to a numpy float32 array of shape (C, T)."""
    data = np.array(aseg.get_array_of_samples()).astype(np.float32) / 32768.0
    if aseg.channels == 1:
        return data[np.newaxis, :]
    return data.reshape(-1, aseg.channels).T


def remove_silence(
    audio: NDArray[np.float32],
    sampling_rate: int,
    mid_sil: int = 300,
    lead_sil: int = 100,
    trail_sil: int = 300,
) -> NDArray[np.float32]:
    """Remove middle silences longer than *mid_sil* ms and trim edge silences.

    Parameters:
        audio: numpy array with shape (C, T).
        sampling_rate: sampling rate of the audio.
        mid_sil: middle-silence threshold in ms (0 to skip).
        lead_sil: kept leading silence in ms.
        trail_sil: kept trailing silence in ms.

    Returns:
        Numpy array with shape (C, T').
    """
    wave = numpy_to_audiosegment(audio, sampling_rate)

    if mid_sil > 0:
        non_silent_segs = split_on_silence(
            wave,
            min_silence_len=mid_sil,
            silence_thresh=-50,
            keep_silence=mid_sil,
            seek_step=10,
        )
        wave = AudioSegment.silent(duration=0)
        for seg in non_silent_segs:
            wave += seg

    wave = remove_silence_edges(wave, lead_sil, trail_sil, -50)

    return audiosegment_to_numpy(wave)


def remove_silence_edges(
    audio: AudioSegment,
    lead_sil: int = 100,
    trail_sil: int = 300,
    silence_threshold: float = -50,
) -> AudioSegment:
    """Remove edge silences, keeping *lead_sil* / *trail_sil* ms."""
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - lead_sil)
    audio = audio[start_idx:]

    audio = audio.reverse()
    start_idx = detect_leading_silence(audio, silence_threshold=silence_threshold)
    start_idx = max(0, start_idx - trail_sil)
    audio = audio[start_idx:]
    audio = audio.reverse()

    return audio


def fade_and_pad_audio(
    audio: NDArray[np.float32],
    pad_duration: float = 0.1,
    fade_duration: float = 0.1,
    sample_rate: int = 24000,
) -> NDArray[np.float32]:
    """Apply fade-in/out and pad with silence to prevent clicks.

    Args:
        audio: numpy array of shape (C, T).
        pad_duration: silence padding duration per side (seconds).
        fade_duration: fade curve duration (seconds).
        sample_rate: audio sampling rate.

    Returns:
        Processed numpy array of shape (C, T_new).
    """
    if audio.shape[-1] == 0:
        return audio

    fade_samples = int(fade_duration * sample_rate)
    pad_samples = int(pad_duration * sample_rate)

    processed = audio.copy()

    if fade_samples > 0:
        k = min(fade_samples, processed.shape[-1] // 2)
        if k > 0:
            fade_in = np.linspace(0, 1, k, dtype=np.float32)[np.newaxis, :]
            processed[..., :k] *= fade_in

            fade_out = np.linspace(1, 0, k, dtype=np.float32)[np.newaxis, :]
            processed[..., -k:] *= fade_out

    if pad_samples > 0:
        silence = np.zeros(
            (processed.shape[0], pad_samples),
            dtype=processed.dtype,
        )
        processed = np.concatenate([silence, processed, silence], axis=-1)

    return processed


def cross_fade_chunks(
    chunks: list[NDArray[np.float32]],
    sample_rate: int,
    silence_duration: float = 0.3,
) -> NDArray[np.float32]:
    """Concatenate audio chunks with silence gaps and cross-fade at boundaries.

    Args:
        chunks: list of numpy arrays, each (C, T).
        sample_rate: audio sample rate.
        silence_duration: total silence gap duration in seconds.

    Returns:
        Merged numpy array (C, T_total).
    """
    if len(chunks) == 1:
        return chunks[0]

    total_n = int(silence_duration * sample_rate)
    fade_n = total_n // 3
    silence_n = fade_n
    merged = chunks[0].copy()

    for chunk in chunks[1:]:
        parts = [merged]

        fout_n = min(fade_n, merged.shape[-1])
        if fout_n > 0:
            w_out = np.linspace(1, 0, fout_n, dtype=np.float32)[np.newaxis, :]
            parts[-1][..., -fout_n:] *= w_out

        parts.append(np.zeros((chunks[0].shape[0], silence_n), dtype=np.float32))

        fade_in = chunk.copy()
        fin_n = min(fade_n, fade_in.shape[-1])
        if fin_n > 0:
            w_in = np.linspace(0, 1, fin_n, dtype=np.float32)[np.newaxis, :]
            fade_in[..., :fin_n] *= w_in

        parts.append(fade_in)
        merged = np.concatenate(parts, axis=-1)

    return merged
