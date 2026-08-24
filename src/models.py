# models.py

from enum import Enum
from typing import Any, Callable, Coroutine, Mapping 
from dataclasses import dataclass, field
from inspect import isawaitable

@dataclass
class Tool:
    name: str
    description: str
    function: Callable

    async def __call__(self, **kwargs: Any) -> Any:
        res = self.function(**kwargs)
        if isawaitable(res):
            return await res
        return res

@dataclass 
class ToolCall:
    name: str
    arguments: Mapping[str, Any]

@dataclass
class ToolResult:
    result:    Any
    tool_name: str

class Role(Enum):
    USER  = 'user'
    AGENT = 'assistant'
    TOOL  = 'tool'

@dataclass
class Message:
    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)

@dataclass
class LLMRequest:
    messages: list[Message | ToolResult]
    tools:    list[Tool]

@dataclass
class LLMResponse:
    message:    Message

