# agent.py

from llm import LLM
from models import ContentEvent, LLMRequest, Message, Role, ThinkingEvent, ToolCall, ToolResult, ToolCallEvent, ToolResultEvent
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
                    case ContentEvent():
                        content.append(event.content)
                    case ToolCallEvent():
                        tools.extend(event.tool_calls)
                    case _:
                        exit() 
                        # raise TODO: Handle this later
                yield event

            self.history.append(Message(Role.AGENT, ''.join(content), ''.join(thinking), tools))
            
            if tools:
                results = await self.execute_toolcalls(tools)
                self.history += results
                yield ToolResultEvent(results)
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

