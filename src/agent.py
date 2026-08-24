# agent.py

from llm import LLM
from models import LLMRequest, Message, Role, ToolCall, ToolResult
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
            response = self.llm.generate(request)

            self.history.append(response.message)

            if response.message.tool_calls:

                self.history += await self.execute_toolcalls(response.message.tool_calls)

            else:
                # No more tool calls, request is likely complete
                yield response.message.content
                break


    async def execute_toolcalls(self, toolcalls: list[ToolCall]) -> list[ToolResult]:

        results = await asyncio.gather(
            *(self.tools[call.name](**call.arguments) for call in toolcalls)
        )

        return [
            ToolResult(result, call.name) 
            for result, call in zip(results, toolcalls)
        ]
