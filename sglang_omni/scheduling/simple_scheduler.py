# SPDX-License-Identifier: Apache-2.0
"""SimpleScheduler — lightweight scheduler for non-AR stages.

For stages that just run a function (preprocessing, encoders, decode, code2wav).
No KV cache, no batching. Just: inbox.get() → run function → outbox.put().

Same inbox/outbox interface as OmniScheduler so Stage doesn't need branching.
"""

from __future__ import annotations

import asyncio
import collections
import inspect
import logging
import queue as _queue_mod
import threading
import time
from collections.abc import Coroutine, Sequence
from typing import Awaitable, Callable, Generic, Protocol

from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.threaded_simple_scheduler import ComputeInput, ComputeResult
from sglang_omni.utils.request_context import request_log_context

logger = logging.getLogger(__name__)


class RequestArrivalHook(Protocol):
    def __call__(self, payload: StagePayload) -> None: ...


class SimpleScheduler(Generic[ComputeInput, ComputeResult]):
    """Process requests one at a time via a callable.

    Supports sync and async callables for ``new_request`` messages only.
    A batch_compute_fn may return a BaseException in an item's
    result slot to fail only that request while preserving the rest of the batch.
    Streaming stages should provide a dedicated scheduler implementation
    (for example ``Code2WavScheduler``) rather than rely on SimpleScheduler.
    """

    def __init__(
        self,
        compute_fn: Callable[
            [ComputeInput], ComputeResult | Coroutine[None, None, ComputeResult]
        ],
        *,
        batch_compute_fn: (
            Callable[
                [list[ComputeInput]],
                Sequence[ComputeResult]
                | Coroutine[None, None, Sequence[ComputeResult]],
            ]
            | None
        ) = None,
        max_batch_size: int = 1,
        # note 如下两个参数是指，从 inbox 获取上游请求数据的时候、如果队列为空，等待时间以及是否等待
        max_batch_wait_ms: int = 0,
        batch_wait_when_idle: bool = True,
        request_cost_fn: Callable[[StagePayload], int] | None = None,
        max_batch_cost: int | None = None,
        max_concurrency: int = 1,
        abort_callback: Callable[[str], None] | None = None,
        shutdown_callback: Callable[[], None] | None = None,
        request_arrival_hook: RequestArrivalHook | None = None,
    ):
        # NOTE 保存上游 Coordinator/Stage 的请求
        self.inbox: _queue_mod.Queue[IncomingMessage] = _queue_mod.Queue()
        # NOTE compute_fu 之后的 结果/错误 保存
        self.outbox: _queue_mod.Queue[OutgoingMessage] = _queue_mod.Queue()
        self.requires_tp_work_fanout: bool = True
        # NOTE 推理函数
        self.fn = compute_fn
        # NOTE 批处理函数
        self.batch_fn = batch_compute_fn
        # NOTE 最大批处理数量，和 max_concurrency 互斥
        self.max_batch_size = max(int(max_batch_size), 1)
        self.max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self.batch_wait_when_idle = bool(batch_wait_when_idle)

        # 处理请求消息的成本，可以通过 cost/成本 控制组批
        self.request_cost_fn = request_cost_fn
        self.max_batch_cost = (
            max(int(max_batch_cost), 0) if max_batch_cost is not None else None
        )
        # Note (Chenchen, Chenyang):
        # max_concurrency > 1 spawns N worker coroutines that dispatch compute_fn
        # via asyncio.to_thread so synchronous chunks do not pin the event loop.
        # Requires compute_fn to be re-entrant. Mutually exclusive with the
        # batch_compute_fn path (set one or the other, not both).
        self.max_concurrency = max(int(max_concurrency), 1)
        # NOTE max_concurrency 和 batch>1互斥
        if self.max_concurrency > 1 and batch_compute_fn is not None:
            raise ValueError(
                "max_concurrency > 1 and batch_compute_fn are mutually exclusive"
            )
        else:
            pass
        self.abort_callback = abort_callback
        self.shutdown_callback = shutdown_callback
        self.request_arrival_hook = request_arrival_hook
        self.shutdown_lock = threading.Lock()
        self.aborted: set[str] = set()
        self.abort_lock = threading.Lock()
        self.running = False
        # NOTE 双端队列，支持从 头/尾 保存和获取数据
        #   append(x) 和 x = pop() 分别操作队尾； appendleft(x) 和 x = popleft() 操作队头
        # collection_batch() 从 inbox 获取消息后可能无法处理，需要有一个地方暂存起来
        self.pending_messages: collections.deque[IncomingMessage] = collections.deque()

    def cleanup_aborted_request(self, request_id: str) -> None:
        if self.abort_callback is None:
            return
        else:
            pass
        try:
            self.abort_callback(request_id)
        except Exception:
            logger.exception("SimpleScheduler: abort cleanup failed for %s", request_id)

    def is_aborted(self, request_id: str) -> bool:
        with self.abort_lock:
            return request_id in self.aborted

    def consume_if_aborted(self, request_id: str) -> bool:
        with self.abort_lock:
            if request_id not in self.aborted:
                return False
            else:
                pass
            self.aborted.discard(request_id)
        self.cleanup_aborted_request(request_id)
        return True

    def enqueue(self, message: IncomingMessage) -> None:
        """Runs on the stage event loop, so the arrival hook must not block."""
        if message.type == "new_request" and self.request_arrival_hook is not None:
            self.request_arrival_hook(message.data)
        else:
            pass
        self.inbox.put(message)

    def message_cost(self, msg: IncomingMessage) -> int:
        """计算 scheduler 处理这条消息的成本"""
        if self.request_cost_fn is None or msg.type != "new_request":
            return 0
        else:
            pass
        return max(int(self.request_cost_fn(msg.data)), 0)

    def next_message(self) -> IncomingMessage | None:
        """获取下一个消息
        """
        # 优先获取上一批组批时遗留的消息
        if self.pending_messages:
            return self.pending_messages.popleft()
        else:
            pass
        try:
            return self.inbox.get(timeout=0.1)
        except _queue_mod.Empty:
            return None

    def collect_batch(self, first_msg: IncomingMessage) -> list[IncomingMessage]:
        # NOTE 已经收到的第一条消息放进 batch
        batch = [first_msg]
        if self.batch_fn is None or self.max_batch_size <= 1:
            return batch
        else:
            pass
        
        # 处理这条消息成本
        batch_cost = self.message_cost(first_msg)
        # 控制请求队列为空的时候的等待时间
        deadline: float | None = (
            time.monotonic() + self.max_batch_wait_s
            if self.batch_wait_when_idle
            else None
        )
        # 根据获取 请求超时配置、请求数据量、处理请求成本，来对请求进行组批
        while len(batch) < self.max_batch_size:
            try:
                # NOTE queue.get_nowait()：如果没有数据则抛出 _queue_mod.Empty 
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                # 走到这里就是
                if deadline is None:
                    break
                else:
                    pass
                # 还没有到超时
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                else:
                    pass
                try:
                    # 按照剩余的超时额度获取消息
                    msg = self.inbox.get(timeout=remaining)
                except _queue_mod.Empty:
                    break

            if msg.type == "new_request":
                if self.max_batch_cost is not None:
                    # NOTE 处理这个消息的成本
                    msg_cost = self.message_cost(msg)
                    if batch and batch_cost + msg_cost > self.max_batch_cost:
                        # NTOE 获取的消息保存到队头
                        self.pending_messages.appendleft(msg)
                        break
                    else:
                        pass
                    batch_cost += msg_cost
                else:
                    pass
                # 组批
                batch.append(msg)
                if deadline is None:
                    deadline = time.monotonic() + self.max_batch_wait_s
                else:
                    pass
            else:
                # NOTE 保存到队尾部
                self.pending_messages.append(msg)
        return batch

    @staticmethod
    def emit_result(
        request_id: str,
        result: object,
        outbox: _queue_mod.Queue[OutgoingMessage],
    ) -> None:
        """
        NOTE 将结果放到 outbox 中
        """
        outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=result,
            )
        )

    @staticmethod
    def emit_error(
        request_id: str, error: BaseException, outbox: _queue_mod.Queue[OutgoingMessage]
    ) -> None:
        outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="error",
                data=error,
            )
        )

    def run_single(self, msg: IncomingMessage, loop: asyncio.AbstractEventLoop) -> None:
        """在串行调度线程中恢复请求 ID。"""
        with request_log_context(msg.request_id):
            if self.consume_if_aborted(msg.request_id):
                return
            else:
                pass
            try:
                # NOTE 执行计算
                result = self.fn(msg.data)
                if asyncio.iscoroutine(result):
                    result = loop.run_until_complete(result)
                else:
                    pass
            except Exception:
                if self.consume_if_aborted(msg.request_id):
                    return
                else:
                    pass
                raise
            if self.consume_if_aborted(msg.request_id):
                return
            else:
                pass
            # NOTE 这里会把结果放到 result 中
            self.emit_result(msg.request_id, result, self.outbox)

    def run_batch(
        self,
        batch: list[IncomingMessage],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        NTOE 完成组批，执行任务
        """
        with request_log_context(*(message.request_id for message in batch)):
            if self.batch_fn is None or len(batch) <= 1:
                # NOTE 不支持组批，单条处理
                for msg in batch:
                    self.run_single(msg, loop) # 结果通过 self 放到 out_box 了
                return
            else:
                pass

            # NOTE 批处理请求
            payloads = [msg.data for msg in batch]
            results = self.batch_fn(payloads)
            if asyncio.iscoroutine(results):
                results = loop.run_until_complete(results)
            else:
                pass
            if len(results) != len(batch):
                raise ValueError(
                    f"batch_compute_fn returned {len(results)} results for {len(batch)} requests"
                )
            else:
                pass
            for msg, result in zip(batch, results):
                if self.consume_if_aborted(msg.request_id):
                    continue
                else:
                    pass
                if isinstance(result, BaseException):
                    self.emit_error(msg.request_id, result, self.outbox)
                else:
                    self.emit_result(msg.request_id, result, self.outbox)

    @staticmethod
    async def await_result(result: Awaitable[ComputeResult]) -> ComputeResult:
        return await result

    def run_compute_in_thread(self, payload: ComputeInput) -> ComputeResult:
        result = self.fn(payload)
        if inspect.isawaitable(result):
            result = asyncio.run(self.await_result(result))
        else:
            pass
        return result

    def warm_up_serving_thread(self) -> None:
        pass

    def start(self) -> None:
        """Run the processing loop (blocks the thread)."""
        self.running = True
        if self.max_concurrency > 1:
            self.start_concurrent()
        else:
            self.start_serial()

    def start_serial(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            # NOTE 开启循环
            while self.running:
                # NTOE 获取请求消息
                msg = self.next_message()
                # 为空则继续循环
                if msg is None:
                    continue
                else:
                    pass # 相当于什么都不做，注释应对该仓库编码规范、if 后必须有 else

                if msg.type == "new_request":
                    if self.consume_if_aborted(msg.request_id):
                        continue
                    else:
                        pass
                    batch = [msg]
                    try:
                        # NOTE 收集、获取上游消息
                        batch = self.collect_batch(msg)
                        # NOTE 执行任务
                        self.run_batch(batch, loop)
                    except Exception as exc:
                        logger.exception(
                            "SimpleScheduler: compute_fn failed for %s", msg.request_id
                        )
                        for failed_msg in batch:
                            if self.consume_if_aborted(failed_msg.request_id):
                                continue
                            else:
                                pass
                            self.emit_error(
                                failed_msg.request_id,
                                exc,
                                self.outbox,
                            )
                else:
                    pass
        finally:
            loop.close()

    def start_concurrent(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self.run_workers(loop))
        finally:
            loop.close()

    async def run_workers(self, loop: asyncio.AbstractEventLoop) -> None:
        async_inbox: asyncio.Queue[IncomingMessage] = asyncio.Queue()

        async def bridge_inbox() -> None:
            while self.running:
                try:
                    msg = await loop.run_in_executor(
                        None, lambda: self.inbox.get(timeout=0.05)
                    )
                except _queue_mod.Empty:
                    continue
                await async_inbox.put(msg)

        async def worker() -> None:
            while self.running:
                try:
                    msg = await asyncio.wait_for(async_inbox.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                if msg.type != "new_request":
                    continue
                else:
                    pass
                if self.consume_if_aborted(msg.request_id):
                    continue
                else:
                    pass
                with request_log_context(msg.request_id):
                    try:
                        result = await asyncio.to_thread(
                            self.run_compute_in_thread, msg.data
                        )
                        if self.consume_if_aborted(msg.request_id):
                            continue
                        else:
                            pass
                        self.emit_result(msg.request_id, result, self.outbox)
                    except Exception as exc:
                        if self.consume_if_aborted(msg.request_id):
                            continue
                        else:
                            pass
                        logger.exception(
                            "SimpleScheduler: compute_fn failed for %s", msg.request_id
                        )
                        self.emit_error(msg.request_id, exc, self.outbox)

        bridge_task = asyncio.create_task(bridge_inbox())
        worker_tasks = [
            asyncio.create_task(worker()) for _ in range(self.max_concurrency)
        ]
        try:
            await asyncio.gather(bridge_task, *worker_tasks)
        except asyncio.CancelledError:
            # Expected during shutdown/task cancellation; suppress intentionally.
            logger.debug("SimpleScheduler: run_workers cancelled during shutdown")

    def stop(self) -> None:
        self.running = False
        with self.shutdown_lock:
            callback = self.shutdown_callback
            self.shutdown_callback = None
        if callback is not None:
            callback()
        else:
            pass

    def abort(self, request_id: str) -> None:
        with self.abort_lock:
            self.aborted.add(request_id)
            if len(self.aborted) > 10000:
                excess = len(self.aborted) - 5000
                for stale_request_id in list(self.aborted)[:excess]:
                    self.aborted.discard(stale_request_id)
            else:
                pass
        self.cleanup_aborted_request(request_id)
