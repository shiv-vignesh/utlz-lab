"""OpenAI wire shapes. A deliberately SMALL subset. Write it yourself, then diff against
vllm/entrypoints/openai/completion/ and chat_completion/protocol.py and list what you left out.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StreamOptions(BaseModel):
    include_usage: bool = False

class SamplingFields(BaseModel):
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None
    stop: list[str] | str | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    # vLLM extensions (outside the OpenAI spec)
    ignore_eos: bool = False
    min_tokens: int = 0

class CompletionRequest(SamplingFields):
    model: str
    prompt: str | list[int]          # token ids accepted: Phase 1-comparable ISL

"""
# <--------- Non-streaming completion Response ------>

- Non-streaming text response. 
- One completed generation or Max Length Reached 
"""

class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Literal["stop", "length", "content_filter"] | None = None

class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage | None = None

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(SamplingFields):
    model: str
    messages: list[ChatMessage]

"""
# ---------- Streaming completion ----------
Streaming text response	One incremental piece of generated text
"""
class CompletionChunkChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: Literal["stop", "length", "abort", "error", "repetition"] | None = None

class CompletionChunk(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChunkChoice]
    usage: Usage | None = None

    # Lab extension: populate only on the final chunk.
    server_timing: dict[str, float] | None = None

"""
---------- Non-streaming chat completion ----------
Non-streaming chat response	One completed assistant message
"""

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatMessageResponse(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessageResponse
    finish_reason: Literal["stop", "length", "content_filter"] | None = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage | None = None


# ---------- Streaming chat completion ----------
class ChatDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatDelta = Field(default_factory=ChatDelta)
    finish_reason: Literal["stop", "length", "content_filter"] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: Usage | None = None

    # Lab extension: populate only on the final chunk.
    server_timing: dict[str, float] | None = None


# ---------- Errors ----------
class ErrorDetail(BaseModel):
    message: str
    type: str
    code: str | int | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail