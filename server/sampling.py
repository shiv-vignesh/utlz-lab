
from __future__ import annotations

from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from server.protocol import SamplingFields


def to_sampling_params(req: SamplingFields) -> SamplingParams:
    return SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
        stop=req.stop,
        min_tokens=req.min_tokens,
        ignore_eos=req.ignore_eos,
        # DELTA: each RequestOutput carries only the new text/tokens (default CUMULATIVE resends
        # everything so far). FINAL_ONLY: one RequestOutput when the request finishes.
        output_kind=RequestOutputKind.DELTA if req.stream else RequestOutputKind.FINAL_ONLY,
    )