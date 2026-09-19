# llm.py
# abstraction for llm calls

import inspect
import itertools
import json
from collections.abc import AsyncGenerator, Callable
from types import UnionType
from typing import Any, Protocol, Union, get_args, get_origin, get_type_hints

import groq
import ollama

from harness.models import (
    AgentEvent,
    ContentEvent,
    LLMRequest,
    LLMResponse,
    Message,
    Role,
    ThinkingEvent,
    Tool,
    ToolCall,
    ToolCallEvent,
    ToolResult,
)


class LLM(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse: ...
    def generate_stream(self, request: LLMRequest) -> AsyncGenerator[AgentEvent, None]: ...


class OllamaLLM(LLM):

    def __init__(self, model: str = 'qwen3.5:9b'):
        self.model = model
        self.client = ollama.AsyncClient()

    async def generate(self, request: LLMRequest) -> LLMResponse:

        response = await self.client.chat(
            model=self.model,
            tools=[tool.function for tool in request.tools],
            messages=[self.to_ollama(message) for message in request.messages],
            think=True
        )

        tool_calls = [
            ToolCall(tool.function.name, tool.function.arguments)
            for tool in response.message.tool_calls or []
        ]

        return LLMResponse(
            Message(Role.AGENT, response.message.content or '', response.message.thinking or '', tool_calls)
        )


    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[AgentEvent, None]:

        response = await self.client.chat(
            model=self.model,
            tools=[tool.function for tool in request.tools],
            messages=[self.to_ollama(message) for message in request.messages],
            think=True,
            stream=True
        )

        async for chunk in response:
            message = chunk.message

            if message.thinking:
                yield ThinkingEvent(message.thinking)
            if message.content:
                yield ContentEvent(message.content)
            if message.tool_calls:
                yield ToolCallEvent(
                    [
                        ToolCall(tool.function.name, tool.function.arguments) 
                        for tool in message.tool_calls
                    ]
                )

            if chunk.done:
                return



    def to_ollama(self, message: Message | ToolResult):

        if isinstance(message, Message):
            result: dict[str, Any] = {
                "role": message.role.value,
                "content": message.content,
                "thinking": message.thinking
            }

            if message.tool_calls:
                result["tool_calls"] = [
                    {
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                    }
                    for call in message.tool_calls
                ]

            return result
        elif isinstance(message, ToolResult):
            return {'role': 'tool', 'content':str(message.result), 'tool_name':message.tool_name}


class GroqLLM(LLM):

    def __init__(self, model: str = 'openai/gpt-oss-120b', api_key: str | None = None):
        self.model = model
        # Without an api_key the client reads GROQ_API_KEY from the environment.
        self.client = groq.AsyncGroq(api_key=api_key)

    async def generate(self, request: LLMRequest) -> LLMResponse:

        response = await self.client.chat.completions.create(**self.params(request))
        message = response.choices[0].message

        tool_calls = [
            ToolCall(call.function.name, json.loads(call.function.arguments or '{}'))
            for call in message.tool_calls or []
        ]

        return LLMResponse(
            Message(Role.AGENT, message.content or '', message.reasoning or '', tool_calls)
        )


    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[AgentEvent, None]:

        stream = await self.client.chat.completions.create(**self.params(request), stream=True)

        # A call can arrive in pieces, keyed by its index, so the calls are only
        # complete once the stream ends.
        calls: dict[int, tuple[str, str]] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.reasoning:
                yield ThinkingEvent(delta.reasoning)
            if delta.content:
                yield ContentEvent(delta.content)
            for part in delta.tool_calls or []:
                name, arguments = calls.get(part.index, ('', ''))
                if part.function:
                    name      += part.function.name or ''
                    arguments += part.function.arguments or ''
                calls[part.index] = (name, arguments)

        if calls:
            yield ToolCallEvent(
                [
                    ToolCall(name, json.loads(arguments or '{}'))
                    for _, (name, arguments) in sorted(calls.items())
                ]
            )


    def params(self, request: LLMRequest) -> dict[str, Any]:

        params: dict[str, Any] = {'model': self.model, 'messages': self.to_groq(request.messages)}
        if request.tools:
            params['tools'] = [to_groq_tool(tool) for tool in request.tools]
        return params


    def to_groq(self, messages: list[Message | ToolResult]) -> list[dict[str, Any]]:

        # Groq matches each result to its call by id. The harness keeps results
        # in call order instead, the way ollama expects, so the ids are made up
        # here: each result takes the oldest id still waiting for one.
        ids = (f'call_{n}' for n in itertools.count())
        waiting: list[str] = []
        converted: list[dict[str, Any]] = []

        for message in messages:
            if isinstance(message, ToolResult):
                converted.append(
                    {'role': 'tool', 'tool_call_id': waiting.pop(0), 'content': str(message.result)}
                )
                continue

            entry: dict[str, Any] = {'role': message.role.value, 'content': message.content}

            if message.tool_calls:
                waiting = [next(ids) for _ in message.tool_calls]
                entry['tool_calls'] = [
                    {
                        'id': call_id,
                        'type': 'function',
                        'function': {
                            'name': call.name,
                            'arguments': json.dumps(call.arguments),
                        },
                    }
                    for call_id, call in zip(waiting, message.tool_calls, strict=True)
                ]

            converted.append(entry)

        return converted


JSON_TYPES = {str: 'string', int: 'integer', float: 'number', bool: 'boolean',
              list: 'array', dict: 'object'}


def to_groq_tool(tool: Tool) -> dict[str, Any]:
    '''Describe a tool as the OpenAI-style function schema Groq takes.'''

    hints = get_type_hints(tool.function)
    properties: dict[str, Any] = {}
    required:   list[str] = []

    for name, param in inspect.signature(tool.function).parameters.items():
        properties[name] = json_type(hints.get(name, Any))
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        'type': 'function',
        'function': {
            'name': tool.name,
            'description': tool.description,
            'parameters': {'type': 'object', 'properties': properties, 'required': required},
        },
    }


def json_type(hint: Any) -> dict[str, Any]:
    '''The JSON schema for one parameter's type hint.'''

    origin = get_origin(hint) or hint
    args = [arg for arg in get_args(hint) if arg is not type(None)]

    # `X | None` only says the parameter is optional, which `required` covers.
    if origin in (Union, UnionType):
        return json_type(args[0]) if len(args) == 1 else {}
    if origin is list and args:
        return {'type': 'array', 'items': json_type(args[0])}
    if origin in JSON_TYPES:
        return {'type': JSON_TYPES[origin]}
    return {}


# provider name -> (implementation, default model)
PROVIDERS: dict[str, tuple[Callable[[str], LLM], str]] = {
    'ollama': (OllamaLLM, 'qwen3.5:9b'),
    'groq':   (GroqLLM, 'openai/gpt-oss-120b'),
}

