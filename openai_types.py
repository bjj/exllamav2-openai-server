from __future__ import annotations
from pydantic import BaseModel

class ChatCompletionsMessage(BaseModel):
    content: str
    role: str
    name: str | None = None

class ChatCompletions(BaseModel):
    messages: list[ChatCompletionsMessage]
    model: str
    frequency_penalty: float = 0.0
    logic_bias: dict | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n: int = 1
    presence_penalty: float = 0.0
    response_format: dict | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    tools: list = []
    tool_choice: str | dict | None = None
    user: str | None = None

class ChatCompletionsResponseMessage(BaseModel):
    content: str | None = None
    tool_calls: list | None = None
    role: str

class ChatCompletionsResponseChoice(BaseModel):
    finish_reason: str  # stop / length / content_filter / tool_calls / function_call
    index: int
    message: ChatCompletionsResponseMessage
    logprobs: None = None

class ChatCompletionsResponseUsage(BaseModel):
    completion_tokens: int = 1
    prompt_tokens: int
    total_tokens: int = 1

class ChatCompletionsResponse(BaseModel):
    id: str
    choices: list[ChatCompletionsResponseChoice] = []
    created: int
    model: str
    system_fingerprint: str = "exllamav2"
    object: str = "chat.completion"
    usage: ChatCompletionsResponseUsage

class ChatCompletionsChunkResponseDelta(BaseModel):
    content: str | None = None
    tool_calls: list | None = None
    role: str



class ChatCompletionsChunkResponse(BaseModel):
    class Choice(BaseModel):
        delta: ChatCompletionsChunkResponseDelta
        finish_reason: str | None = None # stop / length / content_filter / tool_calls / function_call
        index: int
        logprobs: None = None

    id: str
    choices: list[ChatCompletionsChunkResponse.Choice] = []
    created: int
    model: str
    system_fingerprint: str = "exllamav2"
    object: str = "chat.completion.chunk"