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


@pytest.mark.parametrize(
    "guidance_scale,logit_dtype,logit_scale",
    [
        pytest.param(0.0, torch.float32, 1.0, id="no-guidance"),
        pytest.param(0.5, torch.float32, 1.0, id="fractional-guidance"),
        pytest.param(2.0, torch.float32, 1.0, id="default-guidance"),
        pytest.param(6.0, torch.float32, 1.0, id="strong-guidance"),
        pytest.param(2.0, torch.float16, 1.0, id="fp16"),
        pytest.param(2.0, torch.bfloat16, 1.0, id="bf16"),
        pytest.param(2.0, torch.float16, 2000.0, id="fp16-large-logits"),
    ],
)
@pytest.mark.parametrize("class_temperature", [0.0, 0.7])
def test_cfg_matches_log_probability_reference(
    guidance_scale: float,
    logit_dtype: torch.dtype,
    logit_scale: float,
    class_temperature: float,
) -> None:
    """化简须保留原 CFG 的置信度、候选截断、MASK 排除与请求随机状态。"""
    input_generator = torch.Generator().manual_seed(11)
    conditional = torch.randn(2, 3, 31, generator=input_generator)
    unconditional = torch.randn(2, 3, 31, generator=input_generator)
    mask_id = conditional.shape[-1] - 1
    conditional[..., mask_id] = 12.0
    unconditional[..., mask_id] = -8.0
    conditional = (conditional * logit_scale).to(logit_dtype)
    unconditional = (unconditional * logit_scale).to(logit_dtype)
    options = GenerationOptions(
        guidance_scale=guidance_scale, class_temperature=class_temperature
    )
    actual_generator = torch.Generator().manual_seed(19)
    reference_generator = torch.Generator().manual_seed(19)
    global_random_state = torch.random.get_rng_state().clone()
    actual, confidence = predict_tokens(
        conditional, unconditional, options, mask_id, actual_generator
    )
    expected = conditional.float().log_softmax(-1)
    if guidance_scale != 0:
        unconditional_log_probs = unconditional.float().log_softmax(-1)
        expected = (
            expected + guidance_scale * (expected - unconditional_log_probs)
        ).log_softmax(-1)
    expected[..., mask_id] = -float("inf")
    if class_temperature > 0:
        # 31 个候选保留向上取整后的 4 个，按排名屏蔽以独立核对 top-k 截断。
        candidate_indices = expected.argsort(dim=-1, descending=True)
        filtered = expected.clone().scatter(-1, candidate_indices[..., 4:], -float("inf"))
        uniform_noise = torch.rand(
            expected.shape, dtype=expected.dtype, generator=reference_generator
        )
        expected_codes = (
            filtered / class_temperature
            - torch.log(-torch.log(uniform_noise + 1e-10) + 1e-10)
        ).argmax(-1)
    else:
        expected_codes = expected.argmax(-1)
    assert torch.equal(actual, expected_codes)
    assert (actual != mask_id).all()
    assert torch.isfinite(confidence).all()
    torch.testing.assert_close(
        confidence, expected.max(-1).values, rtol=1e-5, atol=1e-5
    )
    assert torch.equal(actual_generator.get_state(), reference_generator.get_state())
    assert torch.equal(torch.random.get_rng_state(), global_random_state)


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
    target_position_indices = torch.tensor([0])
    first = model(ids, audio, mask, target_position_indices=target_position_indices)
    ids[:, :, 1] = 15
    second = model(ids, audio, mask, target_position_indices=target_position_indices)
    assert not torch.allclose(first[:, 0], second[:, 0], atol=1e-7)


@pytest.mark.parametrize("target_frame_counts", [(2,), (2, 5)])
@pytest.mark.parametrize("sampled", [False, True])
def test_target_projection_matches_full_projection(
    model, tokenizer, monkeypatch, target_frame_counts: tuple[int, ...], sampled: bool
) -> None:
    """先取行必须保持每轮 logits 和最终码本，同时减少真实输出头的输入行数。"""
    options = {
        "position_temperature": 0.7 if sampled else 0.0,
        "class_temperature": 0.4 if sampled else 0.0,
    }
    items = [
        make_item(str(index), frames, **options)
        for index, frames in enumerate(target_frame_counts)
    ]
    input_lengths = [
        prepare_inputs(item, model, tokenizer)[0].shape[-1] for item in items
    ]
    backbone_outputs: list[torch.Tensor] = []
    projection_rows: list[int] = []

    def capture_hidden_states(module, arguments, output) -> None:
        """保留当前轮的真实主干输出，供完整投影基准使用。"""
        backbone_outputs[:] = [output.last_hidden_state]

    def capture_projection_rows(module, arguments) -> None:
        """观测真实线性层输入，防止实现退化为全量投影后裁剪。"""
        projection_rows.append(
            arguments[0].numel() // model.config.llm_config.hidden_size
        )

    def compare_target_logits(module, arguments, output) -> None:
        """独立计算完整投影，并按各请求的原始位置切片作为对照。"""
        hidden_states = backbone_outputs[0]
        full_logits = torch.nn.functional.linear(
            hidden_states, model.audio_heads.weight
        )
        full_logits = full_logits.view(
            hidden_states.shape[0],
            hidden_states.shape[1],
            model.config.num_audio_codebook,
            model.config.audio_vocab_size,
        ).permute(0, 2, 1, 3)
        conditional = [
            full_logits[index, :, length - frames : length]
            for index, (length, frames) in enumerate(
                zip(input_lengths, target_frame_counts, strict=True)
            )
        ]
        unconditional = [
            full_logits[len(items) + index, :, :frames]
            for index, frames in enumerate(target_frame_counts)
        ]
        torch.testing.assert_close(
            output,
            torch.cat(conditional + unconditional, dim=1),
            rtol=1e-5,
            atol=1e-6,
        )

    hooks = [
        model.llm.register_forward_hook(capture_hidden_states),
        model.audio_heads.register_forward_pre_hook(capture_projection_rows),
        model.register_forward_hook(compare_target_logits),
    ]
    try:
        actual = sample_batch(items, model, tokenizer, lambda _: False)
    finally:
        for hook in hooks:
            hook.remove()
    assert projection_rows == (
        [2 * sum(target_frame_counts)] * items[0].state.generation.num_step
    )

    target_forward = model.forward

    def full_projection_reference(
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        target_position_indices: torch.Tensor,
    ) -> torch.Tensor:
        """仅在测试中选中全部行，投影后裁剪，复现修改前的计算范围。"""
        all_positions = torch.arange(
            input_ids.shape[0] * input_ids.shape[-1], device=input_ids.device
        )
        full_logits = target_forward(
            input_ids, audio_mask, attention_mask, target_position_indices=all_positions
        )
        return full_logits.index_select(1, target_position_indices)

    monkeypatch.setattr(model, "forward", full_projection_reference)
    expected = sample_batch(
        [
            make_item(str(index), frames, **options)
            for index, frames in enumerate(target_frame_counts)
        ],
        model,
        tokenizer,
        lambda _: False,
    )
    for actual_codes, expected_codes in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_codes, expected_codes, rtol=0, atol=0)


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
