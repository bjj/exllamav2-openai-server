from __future__ import annotations
from pydantic import BaseModel

class ChatCompletions(BaseModel):
    class Message(BaseModel):
        content: str
        role: str
        name: str | None = None
    
    messages: list[ChatCompletions.Message]
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