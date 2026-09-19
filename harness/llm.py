# llm.py
# abstraction for llm calls

from collections.abc import AsyncGenerator
from typing import Any, Protocol

import ollama

from harness.models import (
    AgentEvent,
    ContentEvent,
    LLMRequest,
    LLMResponse,
    Message,
    Role,
    ThinkingEvent,
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

