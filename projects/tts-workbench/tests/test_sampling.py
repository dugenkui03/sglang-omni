"""验证顺序循环、重复概率、选材范围和每个任务独立的抽样状态。"""

import json

import httpx
import pytest
from pydantic import ValidationError

from tts_workbench.config import RunConfig, Settings
from tts_workbench.dataset import Dataset
from tts_workbench.runner import Runner
from tts_workbench.sampling import DialogueSampler
from tts_workbench.storage import Store


def test_sequential_restarts_at_first_line(settings: Settings) -> None:
    """每个新任务从第一句开始，遍历完后按源顺序循环。"""
    config = RunConfig()
    lines = Dataset(settings.data_directory).select(config)
    sampler = DialogueSampler(lines, config)
    assert [sampler.next_line().line_id for _ in range(5)] == [
        "unit_0",
        "unit_1",
        "unit_0",
        "unit_1",
        "unit_0",
    ]
    assert DialogueSampler(lines, config).next_line().line_id == "unit_0"


def test_zero_repeat_covers_every_line_before_reusing(settings: Settings) -> None:
    """0% 不提前重复，未选池耗尽后仍能持续抽样且不修改源顺序。"""
    config = RunConfig(sample_order="random", repeat_probability=0)
    lines = Dataset(settings.data_directory).select(config)
    sampler = DialogueSampler(lines, config)
    sampler.random.seed(7)
    first_pass = [sampler.next_line().line_id for _ in lines]
    assert set(first_pass) == {"unit_0", "unit_1"}
    assert len(set(first_pass)) == len(lines)
    assert all(sampler.next_line().line_id in first_pass for _ in range(20))
    assert [line.line_id for line in lines] == ["unit_0", "unit_1"]


def test_full_repeat_and_fresh_task_state(settings: Settings) -> None:
    """100% 重复首条；新任务重新从全部未选素材中抽样。"""
    config = RunConfig(sample_order="random", repeat_probability=1)
    lines = Dataset(settings.data_directory).select(config)
    sampler = DialogueSampler(lines, config)
    sampler.random.seed(7)
    first = sampler.next_line().line_id
    assert all(sampler.next_line().line_id == first for _ in range(20))
    fresh = DialogueSampler(lines, config)
    fresh.random.seed(1)
    assert fresh.next_line().line_id != first
    assert len(fresh.seen) == 1
    assert len(sampler.seen) == 1


def test_repeat_probability_is_for_the_seen_pool(settings: Settings) -> None:
    """在两池均非空时，重复概率不随已选样本数量或重复次数增大。"""
    config = RunConfig(sample_order="random", repeat_probability=0.35)
    example = Dataset(settings.data_directory).select(config)[0]
    lines = [example.model_copy(update={"line_id": str(i)}) for i in range(5001)]
    sampler = DialogueSampler(lines, config)
    sampler.random.seed(20261018)
    selected = {sampler.next_line().line_id}
    repeats = 0
    for _ in range(5000):
        line_id = sampler.next_line().line_id
        repeats += line_id in selected
        selected.add(line_id)
    assert 0.33 < repeats / 5000 < 0.37


@pytest.mark.parametrize("probability", [0, 0.2, 1])
def test_sampling_stays_in_selected_material(
    settings: Settings, probability: float
) -> None:
    """只选一条素材时，两池边界始终回退到该素材，不引入其他台词。"""
    config = RunConfig(
        sample_order="random", repeat_probability=probability, line_ids=["unit_1"]
    )
    lines = Dataset(settings.data_directory).select(config)
    sampler = DialogueSampler(lines, config)
    assert [sampler.next_line().line_id for _ in range(5)] == ["unit_1"] * 5


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_repeat_probability(probability: float) -> None:
    """拒绝超出范围和非有限的概率，防止错误配置被静默执行。"""
    with pytest.raises(ValidationError):
        RunConfig(sample_order="random", repeat_probability=probability)


async def test_random_dispatch_repeats_failed_selection_and_persists_config(
    settings: Settings, wav_bytes: bytes
) -> None:
    """实际调度器按随机配置发包；失败样本仍属已选池，配置可从 SQLite 读回。"""
    observed: list[str] = []

    async def transport(request: httpx.Request) -> httpx.Response:
        """仅在隔离测试中模拟首条失败与后续成功，不请求真实模型。"""
        observed.append(json.loads(request.content)["input"])
        if len(observed) == 1:
            return httpx.Response(500, json={"error": "synthetic-test-only"})
        else:
            return httpx.Response(200, content=wav_bytes)

    store = Store(settings.results_directory)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        runner = Runner(settings, Dataset(settings.data_directory), store, client)
        run = await runner.start(
            RunConfig(
                sample_order="random",
                repeat_probability=1,
                requests_per_minute=600,
                duration_seconds=0.25,
            )
        )
        await runner.task
        saved = store.get_run(run.id)
        assert saved.status == "completed"
        assert saved.config.sample_order == "random"
        assert saved.config.repeat_probability == 1
        assert len(observed) >= 2
        assert len(set(observed)) == 1
        stats = store.statistics(run.id)
        assert stats.failed == 1
        assert stats.success == len(observed) - 1
    store.close()
