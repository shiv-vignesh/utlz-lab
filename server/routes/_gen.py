"""Shared by completions and chat: request ids, pre-stream validation, the generation loops."""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from vllm import SamplingParams
from vllm.exceptions import VLLMClientError
from vllm.outputs import CompletionOutput

from server import sse
from server.engine import EngineService
from server.protocol import ErrorDetail, ErrorResponse, SamplingFields, Usage
from server.sampling import to_sampling_params
from server.timing import RequestTimeline, TimelineSink

_OPENAI_FINISH = {"stop": "stop", "length": "length"}


def finish_reason(reason: str | None) -> str | None:
    # vLLM also has abort/error/repetition; OpenAI clients only know stop/length/content_filter
    if reason is None:
        return None
    return _OPENAI_FINISH.get(str(reason), "stop")


def request_id(raw: Request) -> str:
    return raw.headers.get("x-request-id") or uuid.uuid4().hex


def error_response(status: int, message: str, type: str = "invalid_request_error") -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(message=message, type=type, code=status))
    return JSONResponse(status_code=status, content=body.model_dump())


def prepare(engine: EngineService, req: SamplingFields, model: str, prompt_ids: list[int]
            ) -> tuple[SamplingParams, None] | tuple[None, JSONResponse]:
    """Everything that can fail with a 4xx, checked before any byte of a stream is sent."""
    if model != engine.model_name:
        return None, error_response(404, f"model '{model}' not served (have '{engine.model_name}')",
                                    "model_not_found")
    max_len = engine.llm.model_config.max_model_len
    if len(prompt_ids) + req.max_tokens > max_len:
        return None, error_response(
            400, f"prompt ({len(prompt_ids)}) + max_tokens ({req.max_tokens}) > max_model_len ({max_len})")
    try:
        return to_sampling_params(req), None
    except (VLLMClientError, ValueError) as e:
        return None, error_response(400, str(e))


async def stream_sse(
    engine: EngineService,
    sink: TimelineSink,
    tl: RequestTimeline,
    prompt_ids: list[int],
    params: SamplingParams,
    include_usage: bool,
    make_chunk: Callable[[CompletionOutput, str | None], object],
    make_tail: Callable[[Usage | None, dict[str, float]], object],
) -> AsyncIterator[str]:
    tl.mark_submit()
    try:
        async for out in engine.stream({"prompt_token_ids": prompt_ids}, params, tl.request_id):
            tl.observe(out)
            c = out.outputs[0]
            if c.text or c.finish_reason is not None:
                yield sse.event(make_chunk(c, finish_reason(c.finish_reason)))
        usage = Usage(prompt_tokens=tl.n_prompt, completion_tokens=tl.n_output,
                      total_tokens=tl.n_prompt + tl.n_output) if include_usage else None
        yield sse.event(make_tail(usage, tl.durations()))
        yield sse.done()
    except asyncio.CancelledError:
        tl.finish_reason = tl.finish_reason or "abort"   # client gone; AsyncLLM.generate aborted it
        raise
    except Exception as e:
        tl.error = repr(e)
        yield sse.error(str(e))
        yield sse.done()
    finally:
        sink.append(tl)


async def generate_once(
    engine: EngineService,
    sink: TimelineSink,
    tl: RequestTimeline,
    prompt_ids: list[int],
    params: SamplingParams,
) -> CompletionOutput:
    tl.mark_submit()
    try:
        final = None
        async for out in engine.stream({"prompt_token_ids": prompt_ids}, params, tl.request_id):
            tl.observe(out)
            final = out
        return final.outputs[0]
    except asyncio.CancelledError:
        tl.finish_reason = tl.finish_reason or "abort"
        raise
    except Exception as e:
        tl.error = repr(e)
        raise
    finally:
        sink.append(tl)
