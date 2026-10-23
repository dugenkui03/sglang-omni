# SPDX-License-Identifier: Apache-2.0
"""检验原生调度器的错误、取消和音频分段交接。"""

import queue
import threading
from concurrent.futures import CancelledError
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sglang_omni.models.omnivoice.audio import AudioCodec, load_codec
from sglang_omni.models.omnivoice.hf_config import GenerationOptions, VocoderOptions
from sglang_omni.models.omnivoice.payload_types import OmniVoiceState
from sglang_omni.models.omnivoice.stages import (
    batch_scheduler,
    decode_batch,
    encode_batch,
    generate_batch,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.serve.openai_errors import generation_error_status_code


def make_payload(request_id: str, frames: list[int]) -> StagePayload:
    """创建已经完成参考编码的请求。"""
    state = OmniVoiceState(
        text="Hello.",
        reference_text="Hi.",
        reference_codes=torch.ones(2, 3, dtype=torch.long),
        reference_rms=0.2,
        seed=7,
        generation=GenerationOptions(num_step=2),
        chunk_texts=["Hello."] * len(frames),
        target_frames=frames,
        vocoder=VocoderOptions(
            postprocess_output=False, pad_duration=0.0, fade_duration=0.0
        ),
    )
    return StagePayload(
        request_id=request_id, request=OmniRequest(inputs="Hello"), data=state.to_dict()
    )


def test_generation_preserves_order_and_chunk_count(model, tokenizer) -> None:
    """不同长短请求和采样设置均能恢复原始请求顺序。"""
    payloads = [make_payload("long", [2, 3]), make_payload("short", [4])]
    payloads[1].data["generation"]["num_step"] = 3
    results = generate_batch(
        payloads, lambda _: False, model=model, tokenizer=tokenizer
    )
    assert [result.request_id for result in results] == ["long", "short"]
    for result, expected in zip(results, ([2, 3], [4])):
        state = OmniVoiceState.from_dict(result.data)
        assert state.reference_codes is None
        assert [codes.shape[-1] for codes in state.generated_codes] == expected
        assert state.completion_tokens == 2 * sum(expected)
        assert state.prompt_tokens > 0


def test_encoder_failure_is_per_request() -> None:
    """编码失败只影响当前输入，成功的 GPU 张量保持原设备。"""
    codec = Mock(spec=AudioCodec)
    codec.model = SimpleNamespace(config=SimpleNamespace(frame_rate=75))
    codec.encode.side_effect = [
        RuntimeError("bad codec input"),
        torch.ones(2, 7, dtype=torch.long),
    ]
    payloads = [make_payload("bad", [2]), make_payload("good", [2])]
    for payload in payloads:
        payload.data["reference_waveform"] = torch.ones(1, 640)
    results = encode_batch(payloads, lambda _: False, codec=codec, codebooks=2)
    assert isinstance(results[0], RuntimeError)
    assert payloads[0].data == {}
    state = OmniVoiceState.from_dict(results[1].data)
    assert state.reference_waveform is None
    assert state.reference_codes.shape == (2, 7)


def test_decoder_groups_equal_lengths_and_restores_request_order() -> None:
    """相同长度合批，音频不得在两个请求或两个分段之间串位。"""
    codec = Mock(spec=AudioCodec)

    def decode(codes: torch.Tensor) -> torch.Tensor:
        """用码本首值标记每个样本，便于发现错位。"""
        return codes[:, :1, :].float().repeat_interleave(320, dim=-1) / 10

    codec.decode.side_effect = decode
    payloads = [make_payload("a", [2, 3]), make_payload("b", [2])]
    payloads[0].data["generated_codes"] = [
        torch.ones(2, 2, dtype=torch.long),
        torch.full((2, 3), 2, dtype=torch.long),
    ]
    payloads[1].data["generated_codes"] = [torch.full((2, 2), 3, dtype=torch.long)]
    results = decode_batch(payloads, lambda _: False, codec=codec)
    assert [call.args[0].shape for call in codec.decode.call_args_list] == [
        torch.Size([2, 2, 2]),
        torch.Size([1, 2, 3]),
    ]
    assert results[0].request_id == "a" and results[1].request_id == "b"
    np.testing.assert_allclose(
        np.frombuffer(results[1].data["audio_waveform"], dtype=np.float32), 0.3
    )
    assert "generated_codes" not in results[0].data
    assert results[0].data["sample_rate"] == 24000


def test_cancelled_result_clears_payload(model, tokenizer) -> None:
    """取消的请求不保留局部 GPU 数据或输出成功结果。"""
    payload = make_payload("cancel", [2])
    result = generate_batch(
        [payload], lambda _: True, model=model, tokenizer=tokenizer
    )[0]
    assert isinstance(result, CancelledError)
    assert payload.data == {}


def test_abort_while_compute_finishes_drops_outbox_result() -> None:
    """覆盖计算完成晚于取消的调度器交接竞态。"""
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()

    def compute(payloads, is_cancelled):
        """等待测试线程发出取消后才返回。"""
        entered.set()
        assert release.wait(3)
        returned.set()
        return payloads

    scheduler = batch_scheduler(compute, torch.device("cpu"), 1, 0)
    worker = threading.Thread(target=scheduler.start, daemon=True)
    worker.start()
    try:
        payload = make_payload("abort", [2])
        scheduler.inbox.put(
            IncomingMessage(request_id="abort", type="new_request", data=payload)
        )
        assert entered.wait(3)
        scheduler.abort("abort")
        release.set()
        assert returned.wait(3)
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        release.set()
        scheduler.stop()
        worker.join(timeout=3)
        assert not worker.is_alive()


def test_codec_instance_shared_and_errors_classified(monkeypatch) -> None:
    """不同 Stage 不重复加载同设备 Codec；参数错误保持 400。"""
    load_codec.cache_clear()
    constructed = Mock()
    monkeypatch.setattr("sglang_omni.models.omnivoice.audio.AudioCodec", constructed)
    try:
        assert load_codec("checkpoint", "cpu") is load_codec("checkpoint", "cpu")
        assert constructed.call_count == 1
    finally:
        load_codec.cache_clear()
    assert (
        generation_error_status_code(ValueError("OmniVoice request: invalid duration"))
        == 400
    )
    assert generation_error_status_code(RuntimeError("OmniVoice codec failed")) == 500
