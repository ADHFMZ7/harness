# models.py

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from inspect import isawaitable
from typing import Any


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
    is_error:  bool = field(default=False)

class Role(Enum):
    USER  = 'user'
    AGENT = 'assistant'
    TOOL  = 'tool'

@dataclass
class Message:
    role:       Role
    content:    str
    thinking:   str = field(default='')
    tool_calls: list[ToolCall] = field(default_factory=list)

@dataclass
class LLMRequest:
    messages: list[Message | ToolResult]
    tools:    list[Tool]

@dataclass
class LLMResponse:
    message:    Message

class AgentEvent:
    id: int

@dataclass
class ThinkingEvent(AgentEvent):
    thinking: str

@dataclass
class ContentEvent(AgentEvent):
    content: str

@dataclass
class ToolCallEvent(AgentEvent):
    tool_calls: list[ToolCall]

@dataclass
class ToolResultEvent(AgentEvent):
    results: list[ToolResult]

