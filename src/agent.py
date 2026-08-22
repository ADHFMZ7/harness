# agent.py

from llm import LLM
from models import LLMRequest, Message, Role, ToolCall, ToolResult
from tools import ToolRegistry

# Agent needs some sort of memory later


class Agent:

    def __init__(self, llm: LLM, tools: ToolRegistry):
        self.llm   = llm
        self.tools = tools

        self.history: list[Message | ToolResult] = []


    def run(self, prompt: str):

        self.history.append(Message(Role.USER, prompt))

        while True:

            request = LLMRequest(self.history, self.tools.get_tools())
            response = self.llm.generate(request)

            self.history.append(response.message)

            if response.message.tool_calls:

                self.history += self.execute_toolcalls(response.message.tool_calls)

            else:
                # No more tool calls, request is likely complete
                yield response.message.content
                break


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
