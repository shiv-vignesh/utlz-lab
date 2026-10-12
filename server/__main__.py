"""python -m server --model Qwen/Qwen2.5-7B-Instruct --served-model-name qwen [engine flags] [--enable-lab-admin]"""
from __future__ import annotations

import uvicorn
import vllm.envs as envs

from server.app import build_app
from server.config import ServerConfig


def main() -> None:
    cfg = ServerConfig.from_cli()
    uvicorn.run(
        build_app(cfg),
        host=cfg.host,
        port=cfg.port,
        workers=1,   # each worker would build its own AsyncLLM on the same GPU
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
    )


if __name__ == "__main__":   # required: AsyncLLM spawns the EngineCore process
    main()
