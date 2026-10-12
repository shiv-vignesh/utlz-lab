from __future__ import annotations

import httpx
from openai import AsyncOpenAI

from client.config import ClientConfig


def openai_client(cfg: ClientConfig) -> AsyncOpenAI:
    http = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=cfg.max_connections, max_keepalive_connections=cfg.max_connections),
        timeout=httpx.Timeout(600, connect=10),
    )
    # retries would turn failures into long latencies
    return AsyncOpenAI(base_url=f"{cfg.base_url}/v1", api_key=cfg.api_key, max_retries=0, http_client=http)


def admin_client(cfg: ClientConfig) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=cfg.base_url, timeout=60,
                             headers={"Authorization": f"Bearer {cfg.api_key}"})


def parse_metrics(text: str, names: tuple[str, ...]) -> dict[str, float]:
    """Prometheus text -> {name: value summed over label sets}."""
    out = dict.fromkeys(names, 0.0)
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name, _, rest = line.partition("{") if "{" in line else line.partition(" ")
        if name in out:
            out[name] += float(line.rsplit(" ", 1)[1])
    return out
