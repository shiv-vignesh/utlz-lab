"""Per-request server timeline (box clock). Engine timestamps are monotonic: only subtract them from each other."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput
    from vllm.v1.metrics.stats import RequestStateStats

    from common.run import Run


@dataclass
class RequestTimeline:
    request_id: str
    t_arrival: float = field(default_factory=time.time)
    t_submit: float | None = None
    t_first: float | None = None
    t_last: float | None = None
    n_prompt: int = 0
    n_output: int = 0
    n_cached: int | None = None
    finish_reason: str | None = None
    error: str | None = None
    engine_stats: RequestStateStats | None = None   # None with --disable-log-stats

    def mark_submit(self) -> None:
        self.t_submit = time.time()

    def observe(self, out: RequestOutput) -> None:
        now = time.time()
        completion = out.outputs[0]
        if completion.token_ids:
            if self.t_first is None:
                self.t_first = now   # first tokens, not first text: detokenizer can hold text back
            self.n_output += len(completion.token_ids)
        self.t_last = now
        if out.prompt_token_ids is not None:
            self.n_prompt = len(out.prompt_token_ids)
        if out.num_cached_tokens is not None:
            self.n_cached = out.num_cached_tokens
        if out.metrics is not None:
            self.engine_stats = out.metrics
        if completion.finish_reason is not None:
            self.finish_reason = completion.finish_reason

    def durations(self) -> dict[str, float]:
        d: dict[str, float] = {}
        if self.t_submit is not None:
            d["pre_engine_s"] = self.t_submit - self.t_arrival
        if self.t_first is not None:
            d["ttft_s"] = self.t_first - self.t_arrival
        if self.t_last is not None:
            d["e2e_s"] = self.t_last - self.t_arrival
        if self.t_first is not None and self.t_last is not None and self.n_output > 1:
            d["decode_s"] = self.t_last - self.t_first
            d["tpot_s"] = d["decode_s"] / (self.n_output - 1)

        # same intervals as vllm/v1/metrics/stats.py update_from_finished_request; 0 = never happened
        s = self.engine_stats
        if s is not None:
            if s.queued_ts and s.scheduled_ts:
                d["engine_queue_s"] = s.scheduled_ts - s.queued_ts
            if s.scheduled_ts and s.first_token_ts:
                d["engine_prefill_s"] = s.first_token_ts - s.scheduled_ts
            if s.first_token_ts and s.last_token_ts:
                d["engine_decode_s"] = s.last_token_ts - s.first_token_ts
            if s.first_token_latency:
                d["engine_ttft_s"] = s.first_token_latency
            d["num_preemptions"] = float(s.num_preemptions)
        return d

    def row(self) -> dict:
        return {
            "request_id": self.request_id,
            "t_arrival": self.t_arrival,
            "t_submit": self.t_submit,
            "t_first": self.t_first,
            "t_last": self.t_last,
            "n_prompt": self.n_prompt,
            "n_output": self.n_output,
            "n_cached": self.n_cached,
            "finish_reason": self.finish_reason,
            "error": self.error,
            **self.durations(),
        }


class TimelineSink:
    def __init__(self) -> None:
        self.run: Run | None = None
        self._rows: list[dict] = []

    @property
    def active(self) -> bool:
        return self.run is not None

    def start(self, run: Run) -> None:
        self.run, self._rows = run, []

    def append(self, tl: RequestTimeline) -> None:
        if self.run is not None:
            self._rows.append(tl.row())

    def stop(self) -> tuple[Run | None, list[dict]]:
        # swap on the event loop so write_csv (in a thread) never sees appends
        run, rows = self.run, self._rows
        self.run, self._rows = None, []
        return run, rows

    @staticmethod
    def write_csv(path: Path, rows: list[dict]) -> int:
        fieldnames: dict[str, None] = {}
        for r in rows:
            fieldnames.update(dict.fromkeys(r))   # engine_* columns can be absent on some rows
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames), restval="")
            w.writeheader()
            w.writerows(rows)
        return len(rows)
