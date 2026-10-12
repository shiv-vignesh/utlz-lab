from __future__ import annotations

import asyncio
import hashlib
import secrets
from contextlib import asynccontextmanager

import vllm.envs as envs
from fastapi import FastAPI
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send
from vllm.entrypoints.serve.instrumentator.metrics import attach_router

from common import recorder
from server.config import ServerConfig
from server.engine import EngineService
from server.routes import admin, chat, completions
from server.routes._gen import error_response
from server.timing import TimelineSink


class ApiKeyMiddleware:
    """Pure ASGI (not BaseHTTPMiddleware, which wraps streaming bodies and hides disconnects).
    Same scheme as vllm/entrypoints/serve/middleware/authenticate.py, plus /admin."""

    GUARDED = ("/v1", "/admin")

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        self.app = app
        self.key_hash = hashlib.sha256(api_key.encode()).digest()

    def _ok(self, headers: Headers) -> bool:
        scheme, _, token = headers.get("authorization", "").partition(" ")
        return scheme.lower() == "bearer" and secrets.compare_digest(
            hashlib.sha256(token.encode()).digest(), self.key_hash)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (scope["type"] == "http" and scope.get("method") != "OPTIONS"
                and scope["path"].startswith(self.GUARDED) and not self._ok(Headers(scope=scope))):
            await error_response(401, "invalid or missing API key", "authentication_error")(scope, receive, send)
            return
        await self.app(scope, receive, send)


def build_app(cfg: ServerConfig) -> FastAPI:
    engine = EngineService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine.start(cfg.engine_args, cfg.served_model_name)   # minutes; nothing is served until this returns

        log_task = None
        if not cfg.engine_args.disable_log_stats:
            async def log_stats():
                while True:
                    await asyncio.sleep(envs.VLLM_LOG_STATS_INTERVAL)
                    await engine.llm.do_log_stats()
            log_task = asyncio.create_task(log_stats())

        try:
            yield
        finally:
            if log_task is not None:
                log_task.cancel()
            if app.state.lab_recorder is not None:
                await asyncio.to_thread(recorder.stop, app.state.lab_recorder)
            await engine.shutdown()   # also stops EngineCore, so nothing is left holding the GPU

    app = FastAPI(lifespan=lifespan)
    app.state.config = cfg
    app.state.engine = engine
    app.state.timeline_sink = TimelineSink()
    app.state.lab_recorder = None

    app.include_router(completions.router)
    app.include_router(chat.router)
    app.include_router(admin.router)
    if cfg.enable_lab_admin:
        app.include_router(admin.lab)
    attach_router(app)   # /metrics from vLLM's Prometheus registry
    if cfg.api_key:
        app.add_middleware(ApiKeyMiddleware, api_key=cfg.api_key)
    return app
