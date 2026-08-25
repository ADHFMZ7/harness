# agent.py

from llm import LLM
from models import ContentEvent, LLMRequest, Message, Role, ThinkingEvent, ToolCall, ToolResult, ToolCallEvent
from tools import ToolRegistry

import asyncio

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
                        yield event
                    case ContentEvent():
                        content.append(event.content)
                        yield event
                    case ToolCallEvent():
                        tools.extend(event.tool_calls)

            self.history.append(Message(Role.AGENT, ''.join(content), tools))
            
            if tools:
                self.history += await self.execute_toolcalls(tools)
            else:
                break


    async def execute_toolcalls(self, toolcalls: list[ToolCall]) -> list[ToolResult]:

        results = await asyncio.gather(
            *(self.tools[call.name](**call.arguments) for call in toolcalls)
        )

        return [
            ToolResult(result, call.name) 
            for result, call in zip(results, toolcalls)
        ]

