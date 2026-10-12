"""Arrival processes. Open loop sends on schedule regardless of completions (no coordinated omission)."""
from __future__ import annotations

import asyncio
import math
import random
import time

import httpx
from openai import AsyncOpenAI

from client.request import RequestTrace, one_request
from client.transport import parse_metrics

RUNNING, WAITING = "vllm:num_requests_running", "vllm:num_requests_waiting"


def _pick(prompts: list, i: int, offset: int):
    return prompts[(offset + i) % len(prompts)]


async def open_loop(client: AsyncOpenAI, model: str, prompts: list, rate: float, n: int, osl: int,
                    run_id: str, offset: int = 0, seed: int = 0) -> list[RequestTrace]:
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        rid = f"{run_id}-r{rate:g}-{i}"
        tasks.append(asyncio.create_task(
            one_request(client, model, rid, f"rate={rate:g}", _pick(prompts, i, offset), osl)))
        if not math.isinf(rate):
            await asyncio.sleep(rng.expovariate(rate))
    return list(await asyncio.gather(*tasks))


async def closed_loop(client: AsyncOpenAI, model: str, prompts: list, users: int, n: int, osl: int,
                      run_id: str, offset: int = 0) -> list[RequestTrace]:
    """`users` workers; each sends its next request when the previous one finishes."""
    counter = iter(range(n))
    traces: list[RequestTrace] = []

    async def user():
        for i in counter:
            rid = f"{run_id}-u{users}-{i}"
            traces.append(await one_request(client, model, rid, f"users={users}", _pick(prompts, i, offset), osl))

    await asyncio.gather(*(user() for _ in range(users)))
    return traces


async def wait_idle(admin: httpx.AsyncClient, timeout: float = 600, poll: float = 0.5) -> float:
    """Block until the engine has nothing running or waiting; returns seconds waited."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = parse_metrics((await admin.get("/metrics")).text, (RUNNING, WAITING))
        if m[RUNNING] == 0 and m[WAITING] == 0:
            return time.time() - t0
        await asyncio.sleep(poll)
    raise TimeoutError(f"engine not idle after {timeout}s")
