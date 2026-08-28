import pytest

from harness.agent import Agent, IterationLimit
from harness.models import (
    ContentEvent,
    LLMRequest,
    Message,
    ToolCall,
    ToolCallEvent,
    ToolResult,
)
from harness.tools import build_registry
from harness.workspace import HostWorkspace


class ScriptedLLM:
    """An LLM that replays a fixed list of event batches, then talks."""

    def __init__(self, *batches):
        self.batches = list(batches)
        self.requests: list[LLMRequest] = []

    def generate(self, request):
        raise NotImplementedError

    def generate_stream(self, request):
        self.requests.append(request)
        if self.batches:
            yield from self.batches.pop(0)
        else:
            yield ContentEvent("done")


class LoopingLLM(ScriptedLLM):
    """An LLM that never stops calling a tool."""

    def generate_stream(self, request):
        self.requests.append(request)
        yield ToolCallEvent([ToolCall("add", {"a": 1, "b": 2})])


@pytest.fixture
def registry(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    return build_registry(HostWorkspace(root))


async def drain(agent, prompt="go"):
    return [event async for event in agent.run(prompt)]


async def test_a_turn_without_tool_calls_ends(registry):
    agent = Agent(ScriptedLLM([ContentEvent("hello")]), registry)

    events = await drain(agent)

    assert [type(e) for e in events] == [ContentEvent]
    assert len(agent.history) == 2


async def test_tool_results_are_fed_back(registry):
    agent = Agent(
        ScriptedLLM([ToolCallEvent([ToolCall("add", {"a": 2, "b": 3})])]),
        registry,
    )

    await drain(agent)

    results = [item for item in agent.history if isinstance(item, ToolResult)]
    assert [r.result for r in results] == [5]


async def test_the_loop_stops_at_the_iteration_cap(registry):
    agent = Agent(LoopingLLM(), registry, max_iterations=3)

    with pytest.raises(IterationLimit, match="3 rounds"):
        await drain(agent)

    assert len(agent.llm.requests) == 3


async def test_history_is_still_usable_after_hitting_the_cap(registry):
    agent = Agent(LoopingLLM(), registry, max_iterations=2)

    with pytest.raises(IterationLimit):
        await drain(agent)

    # Every assistant turn that asked for tools is followed by their results,
    # so the next prompt appends to a well-formed conversation.
    asked = [m for m in agent.history if isinstance(m, Message) and m.tool_calls]
    returned = [item for item in agent.history if isinstance(item, ToolResult)]

    assert len(asked) == len(returned) == 2
    assert isinstance(agent.history[-1], ToolResult)


async def test_a_failing_tool_comes_back_as_an_error_result(registry):
    agent = Agent(
        ScriptedLLM([ToolCallEvent([ToolCall("read_file", {"path": "/etc/passwd"})])]),
        registry,
    )

    await drain(agent)

    result = next(item for item in agent.history if isinstance(item, ToolResult))
    assert result.is_error
    assert "PathOutsideWorkspace" in result.result


async def test_an_unknown_tool_comes_back_as_an_error_result(registry):
    agent = Agent(ScriptedLLM([ToolCallEvent([ToolCall("nope", {})])]), registry)

    await drain(agent)

    result = next(item for item in agent.history if isinstance(item, ToolResult))
    assert result.is_error
    assert "No tool named" in result.result
