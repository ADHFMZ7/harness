# agent.py

from enum import Enum
from dataclasses import dataclass

from llm import LLM
from models import LLMRequest, LLMResponse, Message, Role, ToolCall, ToolResult
from tools import ToolRegistry

# Agent needs some sort of memory later


class Agent:

    def __init__(self, llm: LLM, tools: ToolRegistry):
        self.llm:   LLM          = llm
        self.tools: ToolRegistry = tools

        self.history: list[Message | ToolResult] = []


    def run(self, prompt: str):

        self.history.append(Message(Role.USER, prompt))

        while True:

            request = LLMRequest(self.history, self.tools.get_tools())

            response = self.llm.generate(request)

            self.history.append(response.message)

            if response.message.tool_calls:

                # run those tool calls.
                results = self.execute_toolcalls(response.message.tool_calls) # list[ToolResult]
                for result in results:
                    self.history.append(result)

            else:
                # No more tool calls, request is likely complete
                yield response.message.content
                break

        return

    def execute_toolcalls(self, toolcalls: list[ToolCall]) -> list[ToolResult]:

        results = []
        for call in toolcalls:
         
            tool = self.tools[call.name]
            output = tool(**call.arguments)

            print(f"Executing toolcall: {tool.name}({call.arguments}) = {output}")

            results.append(
                ToolResult(output, call.name)
            )

        return results
