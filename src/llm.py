# llm.py
# abstraction for llm calls

from typing import Coroutine, Generator, Protocol
import ollama

from models import ToolCall, LLMRequest, LLMResponse, Message, Role, ToolResult, ThinkingEvent, ContentEvent, ToolCallEvent, AgentEvent

class LLM(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse: ...
    def generate_stream(self, request: LLMRequest) -> Generator[AgentEvent, None, None]: ...


class OllamaLLM(LLM):

    def __init__(self, model: str = 'qwen3.5:9b'):
        self.model = model

    def generate(self, request: LLMRequest) -> LLMResponse:

        response = ollama.chat(
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


    def generate_stream(self, request: LLMRequest) -> Generator[AgentEvent, None, None]:

        response = ollama.chat(
            model=self.model,
            tools=[tool.function for tool in request.tools],
            messages=[self.to_ollama(message) for message in request.messages],
            think=True,
            stream=True
        )

        for chunk in response:
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
            result = {
                "role": message.role,
                "content": message.content,
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


from tools import registry
if __name__ == '__main__':

    llm = OllamaLLM()

        # request = LLMRequest(self.history, self.tools.get_tools())

    message = 'What is the result of 5432 + 65453?'
    messages: list[Message | ToolResult] = [Message(Role.USER, message)]
    req = LLMRequest(messages, registry.get_tools())

    for event in llm.generate_stream(req):
        print(event)


