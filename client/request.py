"""One streaming request, timestamps on the Mac clock."""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI


@dataclass
class RequestTrace:
    request_id: str
    point: str                                             # e.g. 'rate=2' or 'users=8'
    t_send: float
    t_chunks: list[float] = field(default_factory=list)   # chunks that carried text
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    server_timing: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    def row(self) -> dict:
        r = {
            "request_id": self.request_id,
            "point": self.point,
            "t_send": self.t_send,
            "ok": self.error is None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "n_chunks": len(self.t_chunks),
            "finish_reason": self.finish_reason,
            "error": self.error,
        }
        if self.t_chunks:
            r["client_ttft_s"] = self.t_chunks[0] - self.t_send
            r["client_e2e_s"] = self.t_chunks[-1] - self.t_send
            itl = [b - a for a, b in zip(self.t_chunks, self.t_chunks[1:])]
            if itl:
                r["itl_median_ms"] = 1e3 * statistics.median(itl)
                r["itl_max_ms"] = 1e3 * max(itl)
            if len(itl) >= 2:
                r["itl_p99_ms"] = 1e3 * statistics.quantiles(itl, n=100)[98]
        r.update({f"srv_{k}": v for k, v in self.server_timing.items()})
        # durations from two clocks are comparable; timestamps are not
        if "client_ttft_s" in r and "ttft_s" in self.server_timing:
            r["network_ttft_s"] = r["client_ttft_s"] - self.server_timing["ttft_s"]
        return r


async def one_request(client: AsyncOpenAI, model: str, request_id: str, point: str,
                      prompt: list[int] | str, max_tokens: int) -> RequestTrace:
    tr = RequestTrace(request_id, point, time.time())
    try:
        stream = await client.completions.create(
            model=model, prompt=prompt, max_tokens=max_tokens, temperature=0.0,
            stream=True, stream_options={"include_usage": True},
            extra_body={"ignore_eos": True},
            extra_headers={"X-Request-Id": request_id},
        )
        async for chunk in stream:
            now = time.time()
            if chunk.choices:
                c = chunk.choices[0]
                if c.text:
                    tr.t_chunks.append(now)
                tr.finish_reason = c.finish_reason or tr.finish_reason
            if chunk.usage is not None:
                tr.prompt_tokens = chunk.usage.prompt_tokens
                tr.completion_tokens = chunk.usage.completion_tokens
            extra = chunk.model_extra or {}
            if extra.get("server_timing"):
                tr.server_timing = extra["server_timing"]
    except Exception as e:
        tr.error = repr(e)
    return tr


async def chat_once(client: AsyncOpenAI, model: str, message: str, max_tokens: int) -> None:
    stream = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": message}],
        max_tokens=max_tokens, stream=True, stream_options={"include_usage": True},
    )
    t0, t_first, usage, timing = time.time(), None, None, None
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            t_first = t_first or time.time()
            print(chunk.choices[0].delta.content, end="", flush=True)
        usage = chunk.usage or usage
        timing = (chunk.model_extra or {}).get("server_timing") or timing
    print()
    if t_first:
        srv = f", server ttft {1e3 * timing['ttft_s']:.0f} ms" if timing and "ttft_s" in timing else ""
        print(f"[ttft {1e3 * (t_first - t0):.0f} ms{srv}, {usage.completion_tokens if usage else '?'} tokens]")
