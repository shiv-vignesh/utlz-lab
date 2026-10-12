from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClientConfig:
    base_url: str = os.environ.get("P2_BASE_URL", "http://localhost:8000")   # ssh -L tunnel default
    api_key: str = os.environ.get("P2_API_KEY", "EMPTY")
    model: str = "qwen"                                  # the server's --served-model-name
    tokenizer: str = "Qwen/Qwen2.5-7B-Instruct"          # HF id, only for exact-ISL token ids
    out_base: Path = Path("results/vllm-client")
    max_connections: int = 1024                          # must exceed peak in-flight requests
