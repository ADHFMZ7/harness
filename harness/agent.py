# agent.py

import asyncio

from harness.llm import LLM
from harness.models import (
    ContentEvent,
    LLMRequest,
    Message,
    Role,
    ThinkingEvent,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
)
from harness.tools import ToolRegistry

# Agent needs some sort of memory later

class Agent:

    def __init__(self, llm: LLM, tools: ToolRegistry):
        self.llm   = llm
        self.tools = tools

        self.history: list[Message | ToolResult] = []


    async def run(self, prompt: str):

        self.history.append(Message(Role.USER, prompt))

        while True:

            request = LLMRequest(self.history, self.tools.get_tools())
            stream = self.llm.generate_stream(request)

            thinking = []
            content =  []
            tools =    []

            for event in stream:

                match event:

                    case ThinkingEvent():
                        thinking.append(event.thinking)
                    case ContentEvent():
                        content.append(event.content)
                    case ToolCallEvent():
                        tools.extend(event.tool_calls)
                    case _:
                        # TODO: Handle this later
                        continue
                yield event

            self.history.append(Message(Role.AGENT, ''.join(content), ''.join(thinking), tools))
            
            if tools:
                results = await self.execute_toolcalls(tools)
                self.history += results
                yield ToolResultEvent(results)
            else:
                break


    async def execute_toolcalls(self, toolcalls: list[ToolCall]) -> list[ToolResult]:

        async def call(tc: ToolCall) -> ToolResult:
            try:
                tool = self.tools[tc.name]
            except KeyError:
                names = ", ".join(t.name for t in self.tools.get_tools())
                return ToolResult(f"No tool named {tc.name!r}. Available tools: {names}",
                                  tc.name, is_error=True)
            try:
                return ToolResult(await tool(**tc.arguments), tc.name)
            except Exception as exc:
                return ToolResult(f"{type(exc).__name__}: {exc}", tc.name, is_error=True)

        return await asyncio.gather(*(call(tc) for tc in toolcalls))
