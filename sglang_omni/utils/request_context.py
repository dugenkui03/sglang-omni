"""Scope request identities across asynchronous tasks and scheduler workers."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

request_ids_context: ContextVar[tuple[str, ...]] = ContextVar(
    "sglang_omni_log_request_ids", default=()
)
inference_details_context: ContextVar[bool] = ContextVar(
    "sglang_omni_log_inference_details", default=True
)


@contextmanager
def inference_log_context(enabled: bool) -> Iterator[None]:
    """限定本轮及其嵌套层的详细日志，退出或异常时恢复原范围。"""
    token = inference_details_context.set(enabled and inference_details_context.get())
    try:
        yield
    finally:
        inference_details_context.reset(token)


def should_log_inference_details() -> bool:
    """复用调度器的首轮判断，不额外累计请求轮数。"""
    return inference_details_context.get()


def should_log_transformer_layer(layer_id: int, layer_count: int) -> bool:
    """仅保留首轮第 1、2 和最后一层的细节，layer_id 从 0 开始。"""
    return should_log_inference_details() and layer_id in (0, 1, layer_count - 1)


@contextmanager
def request_log_context(*request_ids: str) -> Iterator[None]:
    """Bind one request or a whole compute batch and restore the previous scope."""
    token = request_ids_context.set(tuple(dict.fromkeys(request_ids)))
    try:
        yield
    finally:
        request_ids_context.reset(token)


def current_request_label() -> str:
    """Keep batch membership explicit and mark non-request work without guessing."""
    request_ids = request_ids_context.get()
    if len(request_ids) == 1:
        return f"request_id={request_ids[0]}"
    elif request_ids:
        return f"request_ids=[{', '.join(request_ids)}]"
    else:
        return "request_id=-"
