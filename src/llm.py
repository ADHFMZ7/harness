# llm.py
# abstraction for llm calls


from typing import Protocol, Any
from dataclasses import dataclass, asdict
from enum import Enum
from tools import ToolRegistry, ToolCall
import ollama

class Role(Enum):
    USER='user'
    AGENT='assistant'

@dataclass
class Message:
    role: Role
    content: str

    def to_form(self, provider: str) -> Any:
        '''Convert messages to format required by provider'''

        if provider == 'ollama':
            return asdict(self)

        else:
            print("Provider not recognized")

@dataclass
class LLMRequest:
    messages: list[Message]
    tools:    ToolRegistry

@dataclass
class LLMResponse:
    message: Message
    toolcalls: list[ToolCall]

class LLM(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse: ...


class OllamaLLM(LLM):

    def __init__(self, model: str = 'qwen3.5:9b', provider: str = 'ollama'):
        self.model = model
        self.provider = provider

    def generate(self, request: LLMRequest) -> LLMResponse:
        response = ollama.chat(
            model=self.model,
            tools=request.tools.get_tools(self.provider),
            messages=[message.to_form('ollama') for message in request.messages], 
            think=True)

        if response.message.tool_calls:
            tools = []
            for tool in response.message.tool_calls:
                tools.append(ToolCall(tool.Function.name, tool.Function.arguments))

        else:
            tools = []

        msg = Message(Role.AGENT, response.message.content or '')
        return LLMResponse(msg, tools)

