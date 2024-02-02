from __future__ import annotations
from pydantic import BaseModel
import typing

# Don't put defaults here or it overrides everything else
class ChatCompletions(BaseModel):
    class Message(BaseModel):
        content: str
        role: str
        name: str | None = None
    
    messages: list[ChatCompletions.Message]
    model: str
    frequency_penalty: float | None = None
    logit_bias: dict | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n: int = 1
    repetition_penalty: float | None = None # openrouter
    presence_penalty: float | None = None
    response_format: dict | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    temperature: float  | None = None
    top_k: int | None = None # openrouter
    top_p: float | None = None
    min_p: float | None = None # openrouter
    top_a: float | None = None # openrouter
    tools: list = []
    tool_choice: str | dict | None = None
    user: str | None = None

class ChatCompletionsResponse(BaseModel):
    class Choice(BaseModel):
        class Message(BaseModel):
            content: str | None = None
            tool_calls: list | None = None
            role: str
            
        finish_reason: str  # stop / length / content_filter / tool_calls / function_call
        index: int
        message: ChatCompletionsResponse.Choice.Message
        logprobs: None = None
    
    class Usage(BaseModel):
        completion_tokens: int = 0
        prompt_tokens: int = 0
        total_tokens: int = 0
        
    id: str
    choices: list[ChatCompletionsResponse.Choice] = []
    created: int
    model: str
    system_fingerprint: str = "exllamav2"
    object: str = "chat.completion"
    usage: ChatCompletionsResponse.Usage

ChatCompletionsResponse.update_forward_refs()
ChatCompletionsResponse.Choice.update_forward_refs()


class ChatCompletionsChunkResponse(BaseModel):
    class Choice(BaseModel):
        class Delta(BaseModel):
            content: str | None = None
            tool_calls: list | None = None
            role: str
            
        delta: ChatCompletionsChunkResponse.Choice.Delta
        finish_reason: str | None = None # stop / length / content_filter / tool_calls / function_call
        index: int
        logprobs: None = None

    id: str
    choices: list[ChatCompletionsChunkResponse.Choice] = []
    created: int
    model: str
    system_fingerprint: str = "exllamav2"
    object: str = "chat.completion.chunk"
    
ChatCompletionsChunkResponse.update_forward_refs()
ChatCompletionsChunkResponse.Choice.update_forward_refs()

class ModelsResponse(BaseModel):
    class Model(BaseModel):
        id: str
        created: int
        object: str = "model"
        onwed_by: str = "system"
        
    object: str = "list"
    data: list[ModelsResponse.Model]
    
ModelsResponse.update_forward_refs()
ModelsResponse.Model.update_forward_refs()

class ErrorResponse(BaseModel):
    class Error(BaseModel):
        message: str
        type: str
        param: typing.Any = None
        code: typing.Any = None
    
    error: ErrorResponse.Error

ErrorResponse.update_forward_refs()
