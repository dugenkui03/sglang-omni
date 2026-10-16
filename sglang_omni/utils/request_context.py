"""Scope request identities across asynchronous tasks and scheduler workers."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

request_ids_context: ContextVar[tuple[str, ...]] = ContextVar(
    "sglang_omni_log_request_ids", default=()
)


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
