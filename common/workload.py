"""Prompts -> exact-ISL token ids, tokenized like Phase 1 (raw text, no chat template)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Workload:
    prompts: list[str]
    token_ids: list[list[int]]
    isl: int
    osl: int


def load_prompts(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build(path: str | Path, tokenizer, isl: int, osl: int) -> Workload:
    prompts = load_prompts(path)
    enc = tokenizer(prompts, truncation=True, max_length=isl)   # same call as Phase 1, minus padding
    token_ids = [list(ids) for ids in enc["input_ids"]]
    short = [i for i, ids in enumerate(token_ids) if len(ids) != isl]
    if short:
        raise ValueError(f"{len(short)} prompts shorter than isl={isl} (first: line {short[0] + 1})")
    return Workload(prompts, token_ids, isl, osl)


def batches(seq: list, size: int, drop_last: bool = True):
    for i in range(0, len(seq), size):
        chunk = seq[i:i + size]
        if drop_last and len(chunk) < size:
            return
        yield chunk
