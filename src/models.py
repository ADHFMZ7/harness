# models.py

from enum import Enum
from typing import Any, Callable, Mapping 
from dataclasses import dataclass, field

@dataclass
class Tool:
    name: str
    description: str
    function: Callable

    def __call__(self, **kwargs: Any) -> Any:
        return self.function(**kwargs)

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

