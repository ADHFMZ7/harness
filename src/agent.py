# agent.py

from enum import Enum
from dataclasses import dataclass

from llm import LLM, LLMRequest, LLMResponse, Message, Role
from tools import ToolRegistry

# Agent needs some sort of memory later


class Agent:

    def __init__(self, llm: LLM, tools: ToolRegistry):
        self.llm = llm
        self.tools = tools

        self.history: list[Message] = []


    def run(self, prompt: str):

        self.history.append(Message(Role.USER, prompt))

        while True:

            message = LLMRequest(self.history, self.tools)

            resp: LLMResponse = self.llm.generate(message)
            self.history.append(resp.message)

            if resp.tool_calls:
                # run those tool calls.
                self.tools[]

            else:
                yield resp.message.content
                break

            # token to yield execution? For now just break

        return
