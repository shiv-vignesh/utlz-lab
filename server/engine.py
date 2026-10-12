from __future__ import annotations

from collections.abc import AsyncIterator

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.outputs import RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError

from fastapi import HTTPException
from fastapi.responses import JSONResponse

class EngineService:
    def __init__(self):
        self.llm: AsyncLLM | None = None
        self.tokenizer = None
        self.model_name: str = None

    def start(self, engine_args: AsyncEngineArgs, served_model_name:str | None = None) -> None:
        self.llm = AsyncLLM.from_engine_args(engine_args=engine_args)
        self.tokenizer = self.llm.get_tokenizer()
        self.model_name = served_model_name or engine_args.model

    async def stream(self, prompt, params: SamplingParams, request_id: str) -> AsyncIterator[RequestOutput]:
        assert self.llm is not None
        async for out in self.llm.generate(prompt=prompt, sampling_params=params, request_id=request_id):
            yield out

    async def abort(self, request_id: str) -> None:
        assert self.llm is not None
        await self.llm.abort(request_id)

    async def health(self) -> HTTPException | JSONResponse:

        assert self.llm is not None
        try:
            await self.llm.check_health()
            return JSONResponse(
                status_code=200,
                content={"status": "healthy"}
            )

        except EngineDeadError:
            raise HTTPException(status_code=503, detail="vLLM Async Engine is Dead")

    async def shutdown(self) -> None:
        if self.llm is not None:
            self.llm.shutdown()
