"""Server-Sent Events framing. The whole OpenAI streaming protocol is these three functions."""
from __future__ import annotations

import json
from pydantic import BaseModel

from server.protocol import ErrorDetail, ErrorResponse

def event(payload: BaseModel | dict) -> str:
    body = payload.model_dump_json(exclude_none=True) if isinstance(payload, BaseModel) else json.dumps(payload)
    return f"data: {body}\n\n"


def done() -> str:
    return "data: [DONE]\n\n"


def error(message: str, code: int = 500, type: str = "server_error") -> str:
    return event(ErrorResponse(error=ErrorDetail(message=message, type=type, code=code)))