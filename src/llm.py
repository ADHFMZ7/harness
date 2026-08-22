# llm.py
# abstraction for llm calls


from typing import Protocol, Mapping
from dataclasses import asdict
from models import ToolCall, LLMRequest, LLMResponse, Message, Role, ToolResult
import ollama


class LLM(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse: ...


class OllamaLLM(LLM):

    def __init__(self, model: str = 'qwen3.5:9b'):
        self.model = model

    def generate(self, request: LLMRequest) -> LLMResponse:
        # TODO: Convert tools to ollama version?


        response = ollama.chat(
            model=self.model,
            tools=[tool.function for tool in request.tools],
            messages=[self.asdict(message) for message in request.messages] ,
            think=True
        )

        if response.message.tool_calls:
            tools = []
            for tool in response.message.tool_calls:
                tools.append(ToolCall(tool.function.name, tool.function.arguments))

        else:
            tools = []

        msg = Message(Role.AGENT, response.message.content or '')
        return LLMResponse(msg, tools)

    def asdict(self, message: Message | ToolResult) -> Mapping[str, str]:
        if isinstance(message, Message):
            return asdict(message)
        elif isinstance(message, ToolResult):
            return {'role': 'tool', 'content':str(message.result), 'tool_name':message.tool_name}
        
