"""Keep HTTP request identities active through response sending and access logs."""

import uuid

from fastapi import FastAPI
from starlette.datastructures import MutableHeaders
from starlette.types import Message, Receive, Scope, Send

from sglang_omni.utils.request_context import request_log_context


class RequestLoggingApp(FastAPI):
    """Scope the outer ASGI call, including FastAPI's error response middleware."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Keep one server-generated identity until the response has been sent."""
        if scope["type"] != "http":
            await super().__call__(scope, receive, send)
        else:
            if scope["path"] == "/v1/audio/speech":
                prefix = "speech"
            elif scope["path"] == "/v1/audio/speech/batch":
                prefix = "speech-batch"
            else:
                prefix = "http"
            request_id = f"{prefix}-{uuid.uuid4()}"
            scope.setdefault("state", {})["request_id"] = request_id

            async def send_with_request_id(message: Message) -> None:
                """Expose the identity also for validation errors and streamed audio."""
                if message["type"] == "http.response.start":
                    MutableHeaders(scope=message)["X-Request-Id"] = request_id
                else:
                    pass
                await send(message)

            with request_log_context(request_id):
                await super().__call__(scope, receive, send_with_request_id)
