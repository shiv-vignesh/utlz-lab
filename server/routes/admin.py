"""/health, /v1/models always on; /admin/runs only with --enable-lab-admin."""
from __future__ import annotations

import asyncio
import time

import vllm
from fastapi import APIRouter, Request

from common import recorder
from common.run import finish_run, new_run
from server.routes._gen import error_response
from server.timing import TimelineSink

router = APIRouter()


@router.get("/health")
async def health(raw: Request):
    return await raw.app.state.engine.health()


@router.get("/v1/models")
async def models(raw: Request):
    engine = raw.app.state.engine
    return {
        "object": "list",
        "data": [{
            "id": engine.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "utlz-lab",
            "max_model_len": engine.llm.model_config.max_model_len,
        }],
    }


lab = APIRouter(prefix="/admin")


def engine_meta(raw: Request) -> dict:
    llm = raw.app.state.engine.llm
    cfg = llm.vllm_config
    return {
        "vllm_version": vllm.__version__,
        "engine_args": vars(raw.app.state.config.engine_args),
        "num_gpu_blocks": cfg.cache_config.num_gpu_blocks,
        "block_size": cfg.cache_config.block_size,
        "max_model_len": cfg.model_config.max_model_len,
        "max_num_seqs": cfg.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
    }


@lab.post("/runs")
async def start_run(body: dict, raw: Request):
    state = raw.app.state
    if state.timeline_sink.active:
        return error_response(409, f"run {state.timeline_sink.run.run_id} already active")

    # blocking (mkdir, pip freeze, nvidia-smi, recorder warm-up): keep it off the event loop
    run = await asyncio.to_thread(new_run, "vllm-serve", body.get("exp", "mac-client"), body)
    try:
        rec = await asyncio.to_thread(recorder.start, run.path("utlz.csv"), run.run_id)
    except RuntimeError as e:
        await asyncio.to_thread(finish_run, run, failed=True, error=str(e))
        return error_response(503, str(e), "server_error")

    run.meta["engine"] = engine_meta(raw)
    await asyncio.to_thread(run.save_meta)
    state.lab_recorder = rec
    state.timeline_sink.start(run)
    return {"run_id": run.run_id}


@lab.post("/runs/{run_id}/stop")
async def stop_run(run_id: str, raw: Request):
    state = raw.app.state
    sink: TimelineSink = state.timeline_sink
    if not sink.active or sink.run.run_id != run_id:
        return error_response(404, f"run {run_id} is not the active run")

    run, rows = sink.stop()
    rec, state.lab_recorder = state.lab_recorder, None
    await asyncio.to_thread(recorder.stop, rec)
    n = await asyncio.to_thread(TimelineSink.write_csv, run.path("server_requests.csv"), rows)
    await asyncio.to_thread(finish_run, run, requests=n)
    return {"run_id": run_id, "requests": n, "dir": str(run.path(""))}
