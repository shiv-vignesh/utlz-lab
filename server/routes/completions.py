from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from vllm.entrypoints.serve.utils.api_utils import with_cancellation

from server.protocol import (
    CompletionChoice,
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    CompletionResponse,
    Usage,
)
from server.routes._gen import error_response, finish_reason, generate_once, prepare, request_id, stream_sse
from server.timing import RequestTimeline

router = APIRouter()


@router.post("/v1/completions")
@with_cancellation
async def completions(req: CompletionRequest, raw_request: Request):
    engine = raw_request.app.state.engine
    sink = raw_request.app.state.timeline_sink
    rid = request_id(raw_request)
    tl = RequestTimeline(rid)

    prompt_ids = req.prompt if isinstance(req.prompt, list) else engine.tokenizer.encode(req.prompt)
    params, err = prepare(engine, req, req.model, prompt_ids)
    if err is not None:
        return err

    cid, created = f"cmpl-{rid}", int(time.time())

    if req.stream:
        include_usage = bool(req.stream_options and req.stream_options.include_usage)

        def chunk(c, reason):
            return CompletionChunk(id=cid, created=created, model=engine.model_name,
                                   choices=[CompletionChunkChoice(text=c.text, finish_reason=reason)])

        def tail(usage, timing):
            return CompletionChunk(id=cid, created=created, model=engine.model_name, choices=[],
                                   usage=usage, server_timing=timing)

        return StreamingResponse(
            stream_sse(engine, sink, tl, prompt_ids, params, include_usage, chunk, tail),
            media_type="text/event-stream",
        )

    try:
        c = await generate_once(engine, sink, tl, prompt_ids, params)
    except Exception as e:
        return error_response(500, str(e), "server_error")
    return CompletionResponse(
        id=cid, created=created, model=engine.model_name,
        choices=[CompletionChoice(text=c.text, finish_reason=finish_reason(c.finish_reason))],
        usage=Usage(prompt_tokens=tl.n_prompt, completion_tokens=tl.n_output,
                    total_tokens=tl.n_prompt + tl.n_output),
    )
