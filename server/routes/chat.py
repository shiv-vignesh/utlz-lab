from __future__ import annotations

import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from vllm.entrypoints.serve.utils.api_utils import with_cancellation

from server.protocol import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatDelta,
    ChatMessageResponse,
    Usage,
)
from server.routes._gen import error_response, finish_reason, generate_once, prepare, request_id, stream_sse
from server.timing import RequestTimeline

router = APIRouter()


def render(tokenizer, req: ChatCompletionRequest) -> list[int]:
    # tokenize=False then encode: transformers 5 returns a BatchEncoding for tokenize=True.
    # The template already contains the special tokens, so don't add them again.
    text = tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages], add_generation_prompt=True, tokenize=False)
    return tokenizer.encode(text, add_special_tokens=False)


@router.post("/v1/chat/completions")
@with_cancellation
async def chat_completions(req: ChatCompletionRequest, raw_request: Request):
    engine = raw_request.app.state.engine
    sink = raw_request.app.state.timeline_sink
    rid = request_id(raw_request)
    tl = RequestTimeline(rid)

    prompt_ids = render(engine.tokenizer, req)
    params, err = prepare(engine, req, req.model, prompt_ids)
    if err is not None:
        return err

    cid, created = f"chatcmpl-{rid}", int(time.time())

    if req.stream:
        include_usage = bool(req.stream_options and req.stream_options.include_usage)
        first = True

        def chunk(c, reason):
            nonlocal first
            delta = ChatDelta(role="assistant" if first else None, content=c.text or None)
            first = False
            return ChatCompletionChunk(id=cid, created=created, model=engine.model_name,
                                       choices=[ChatCompletionChunkChoice(delta=delta, finish_reason=reason)])

        def tail(usage, timing):
            return ChatCompletionChunk(id=cid, created=created, model=engine.model_name, choices=[],
                                       usage=usage, server_timing=timing)

        return StreamingResponse(
            stream_sse(engine, sink, tl, prompt_ids, params, include_usage, chunk, tail),
            media_type="text/event-stream",
        )

    try:
        c = await generate_once(engine, sink, tl, prompt_ids, params)
    except Exception as e:
        return error_response(500, str(e), "server_error")
    return ChatCompletionResponse(
        id=cid, created=created, model=engine.model_name,
        choices=[ChatCompletionChoice(message=ChatMessageResponse(content=c.text),
                                      finish_reason=finish_reason(c.finish_reason))],
        usage=Usage(prompt_tokens=tl.n_prompt, completion_tokens=tl.n_output,
                    total_tokens=tl.n_prompt + tl.n_output),
    )
