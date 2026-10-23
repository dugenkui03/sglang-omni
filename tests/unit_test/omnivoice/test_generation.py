# SPDX-License-Identifier: Apache-2.0
"""检验双向注意力、去掩码、随机隔离和长文本分段。"""

from concurrent.futures import CancelledError

import pytest
import torch

from sglang_omni.models.omnivoice.generation import (
    SamplingItem,
    pack_batch,
    plan_chunks,
    predict_tokens,
    prepare_inputs,
    sample_batch,
    unmask_schedule,
)
from sglang_omni.models.omnivoice.hf_config import GenerationOptions
from sglang_omni.models.omnivoice.payload_types import OmniVoiceState


def make_item(
    request_id: str, frames: int, seed: int = 3, **options: float | int
) -> SamplingItem:
    """不同请求共享配置，但分别持有随机状态和参考码本。"""
    state = OmniVoiceState(
        text="Hello.",
        reference_text="Hi.",
        language="en",
        reference_codes=torch.tensor([[1, 2, 3], [2, 3, 4]]),
        generation=GenerationOptions(num_step=4, **options),
        seed=seed,
    )
    return SamplingItem(
        request_id=request_id,
        state=state,
        text=state.text,
        target_frames=frames,
        generator=torch.Generator().manual_seed(seed),
    )


@pytest.mark.parametrize("frames,codebooks,steps", [(1, 8, 32), (15, 2, 4), (80, 8, 1)])
def test_schedule_finishes_every_codebook(
    frames: int, codebooks: int, steps: int
) -> None:
    """步数超过 token 数时允许空轮，末轮仍不能留下 mask。"""
    counts = unmask_schedule(frames, codebooks, GenerationOptions(num_step=steps))
    assert len(counts) == steps
    assert min(counts) >= 0
    assert sum(counts) == frames * codebooks


def test_padding_does_not_leak_between_lengths() -> None:
    """有效区域是双向注意力，padding 不作为任何有效 query 的 key。"""
    inputs = [
        (torch.ones(2, length, dtype=torch.long), torch.ones(length, dtype=torch.bool))
        for length in (4, 7)
    ]
    ids, audio, attention = pack_batch(inputs, [2, 3], 8)
    assert ids.shape == (4, 2, 7)
    assert attention.shape == (4, 1, 7, 7)
    assert attention[0, 0, :4, :4].all()
    assert not attention[0, 0, :4, 4:].any()
    assert not attention[2, 0, :2, 2:].any()
    assert attention[2, 0, 2:, 2:].diag().all()
    assert audio[2].sum() == 2


def test_cfg_uses_log_probabilities_and_excludes_mask() -> None:
    """与明确的 CFG 数学表达式比较，mask 即使最大也不可采样。"""
    conditional = torch.tensor([[[1.0, 3.0, 20.0]]])
    unconditional = torch.tensor([[[4.0, 0.0, 3.0]]])
    options = GenerationOptions(guidance_scale=2.0)
    actual, confidence = predict_tokens(
        conditional, unconditional, options, 2, torch.Generator()
    )
    expected = (
        3 * conditional.log_softmax(-1) - 2 * unconditional.log_softmax(-1)
    ).log_softmax(-1)
    expected[..., 2] = -float("inf")
    assert torch.equal(actual, expected.argmax(-1))
    torch.testing.assert_close(confidence, expected.max(-1).values)


def test_iteration_reproducible_and_preserves_global_rng(model, tokenizer) -> None:
    """固定请求 seed 的结果与外部随机调用无关。"""
    before = torch.random.get_rng_state().clone()
    first = sample_batch([make_item("a", 6)], model, tokenizer, lambda _: False)[0]
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(19)
    second = sample_batch([make_item("a", 6)], model, tokenizer, lambda _: False)[0]
    assert torch.equal(first, second)
    assert first.shape == (2, 6)
    assert ((first >= 0) & (first < 8)).all()


def test_batch_matches_separate_requests(model, tokenizer) -> None:
    """批次 padding 不改变各请求的确定性结果。"""
    options = {"position_temperature": 0.0}
    batched = sample_batch(
        [make_item("a", 2, **options), make_item("b", 5, **options)],
        model,
        tokenizer,
        lambda _: False,
    )
    for index, frames in enumerate((2, 5)):
        separate = sample_batch(
            [make_item(str(index), frames, **options)],
            model,
            tokenizer,
            lambda _: False,
        )[0]
        assert torch.equal(batched[index], separate)


def test_model_attention_is_bidirectional(model) -> None:
    """改变后方 token 必须能影响前方位置，防止误用因果掩码。"""
    ids = torch.tensor([[[13, 14], [13, 14]]])
    audio = torch.zeros(1, 2, dtype=torch.bool)
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    first = model(ids, audio, mask)
    ids[:, :, 1] = 15
    second = model(ids, audio, mask)
    assert not torch.allclose(first[:, :, 0], second[:, :, 0], atol=1e-7)


def test_cancellation_returns_error_not_partial_audio(model, tokenizer) -> None:
    """同批一个取消不会使另一个请求丢失。"""
    results = sample_batch(
        [make_item("cancel", 2), make_item("keep", 4)],
        model,
        tokenizer,
        lambda rid: rid == "cancel",
    )
    assert isinstance(results[0], CancelledError)
    assert isinstance(results[1], torch.Tensor)
    assert (results[1] != 8).all()


def test_duration_overrides_speed_and_chunks_preserve_text() -> None:
    """显式时长决定目标帧数；长文本按官方标点策略切分。"""
    state = make_item("a", 2).state
    state.speed = 2.0
    state.generation = GenerationOptions(duration=2.5)
    plan_chunks(state, 75)
    assert state.target_frames == [187]
    state.text = "One sentence. Another sentence. " * 12
    state.generation = GenerationOptions(duration=90.0)
    plan_chunks(state, 75)
    assert len(state.chunk_texts) > 1
    assert "".join(state.chunk_texts).replace(" ", "") == state.text.replace(" ", "")
    assert all(frames > 0 for frames in state.target_frames)


def test_oversized_prompt_does_not_fail_valid_batch_member(model, tokenizer) -> None:
    """无效长输入仅失败自身，另一请求仍完成推理。"""
    oversized = make_item("long", 600)
    valid = make_item("short", 2)
    results = sample_batch([oversized, valid], model, tokenizer, lambda _: False)
    assert isinstance(results[0], ValueError)
    assert isinstance(results[1], torch.Tensor)


def test_context_limit_is_checked_before_target_allocation(
    model, tokenizer, monkeypatch
) -> None:
    """过大的 duration 不能先分配巨量 Tensor 再检查上下文。"""
    item = make_item("too-long", 10**9)

    def unexpected_allocation(*args, **kwargs):
        """无效请求不能分配目标掩码张量。"""
        raise AssertionError("target allocation happened before context check")

    monkeypatch.setattr(torch, "full", unexpected_allocation)
    with pytest.raises(ValueError, match="context length"):
        prepare_inputs(item, model, tokenizer)
