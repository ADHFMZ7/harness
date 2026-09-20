import asyncio
import itertools
import json

import httpx
import ollama
import pytest

from harness.core.llm import GroqLLM, OllamaLLM, ProviderError, to_groq_tool
from harness.core.models import (
    ContentEvent,
    LLMRequest,
    Message,
    Role,
    ToolCall,
    ToolResult,
)
from harness.core.tools import build_registry
from harness.core.workspace import HostWorkspace

CHUNKS = ["one ", "two ", "three"]


def line(content: str, done: bool) -> bytes:
    message = {"role": "assistant", "content": content}
    return json.dumps({"message": message, "done": done}).encode() + b"\n"


async def slow_chat(request: httpx.Request) -> httpx.Response:
    """Stream a reply the way ollama does, one JSON line per chunk, pausing between them."""

    async def body():
        for chunk in CHUNKS:
            await asyncio.sleep(0.01)
            yield line(chunk, done=False)
        yield line("", done=True)

    return httpx.Response(200, content=body())


async def test_streaming_leaves_the_event_loop_free():
    llm = OllamaLLM("fake")
    llm.client = ollama.AsyncClient(transport=httpx.MockTransport(slow_chat))

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0)
            ticks += 1

    beating = asyncio.create_task(heartbeat())
    events, ticks_at = [], [0]
    try:
        request = LLMRequest([Message(Role.USER, "hi")], [])
        async for event in llm.generate_stream(request):
            events.append(event)
            ticks_at.append(ticks)
    finally:
        beating.cancel()

    assert events == [ContentEvent(chunk) for chunk in CHUNKS]
    # The heartbeat must get a turn while each chunk is on its way. A blocking
    # client stalls the whole loop instead, and the count stops moving.
    assert all(a < b for a, b in itertools.pairwise(ticks_at)), f"loop stalled: {ticks_at}"


def test_groq_results_answer_the_calls_in_order():
    history = [
        Message(Role.USER, "go"),
        Message(Role.AGENT, "", tool_calls=[ToolCall("add", {"a": 1, "b": 2}),
                                            ToolCall("add", {"a": 3, "b": 4})]),
        ToolResult(3, "add"),
        ToolResult(7, "add"),
        Message(Role.AGENT, "", tool_calls=[ToolCall("add", {"a": 3, "b": 7})]),
        ToolResult(10, "add"),
    ]

    messages = GroqLLM(api_key="unused").to_groq(history)

    calls   = [call["id"] for m in messages for call in m.get("tool_calls", [])]
    answers = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert answers == calls
    assert len(set(calls)) == 3


def test_groq_schema_requires_only_parameters_without_defaults(tmp_path):
    read_file = build_registry(HostWorkspace(tmp_path))["read_file"]

    parameters = to_groq_tool(read_file)["function"]["parameters"]

    assert parameters["required"] == ["path"]
    assert parameters["properties"] == {
        "path": {"type": "string"},
        "start_line": {"type": "integer"},
        "end_line": {"type": "integer"},
    }


async def test_ollama_sends_the_context_size_and_thinking_setting():
    sent = []

    async def chat(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return await slow_chat(request)

    llm = OllamaLLM("fake", num_ctx=16384, think=False)
    llm.client = ollama.AsyncClient(transport=httpx.MockTransport(chat))

    request = LLMRequest([Message(Role.USER, "hi")], [])
    [event async for event in llm.generate_stream(request)]

    assert sent[0]["options"] == {"num_ctx": 16384}
    assert sent[0]["think"] is False


def test_a_missing_api_key_raises_a_provider_error(monkeypatch):
    # The front-ends catch this. If it came back as groq's own error type they
    # would each have to import groq to name it.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ProviderError) as caught:
        GroqLLM()

    assert "api_key" in str(caught.value)
