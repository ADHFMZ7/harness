import pytest

from harness.core.agent import Agent, IterationLimit
from harness.core.models import (
    ContentEvent,
    LLMRequest,
    Message,
    Role,
    ToolCall,
    ToolCallEvent,
    ToolResult,
)
from harness.core.tools import build_registry
from harness.core.workspace import HostWorkspace


class ScriptedLLM:
    """An LLM that replays a fixed list of event batches, then talks."""

    def __init__(self, *batches):
        self.batches = list(batches)
        self.requests: list[LLMRequest] = []

    async def generate(self, request):
        raise NotImplementedError

    async def generate_stream(self, request):
        self.requests.append(request)
        if self.batches:
            # async generators cannot use `yield from`
            for event in self.batches.pop(0):
                yield event
        else:
            yield ContentEvent("done")


class LoopingLLM(ScriptedLLM):
    """An LLM that never stops calling a tool."""

    async def generate_stream(self, request):
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
    # The roles rather than the count, so it says what the history holds.
    assert [m.role for m in agent.history] == [Role.SYSTEM, Role.USER, Role.AGENT]


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


async def test_the_system_prompt_opens_the_history(registry):
    agent = Agent(ScriptedLLM(), registry, system="YOU ARE THE SCOUT")

    assert agent.history[0] == Message(Role.SYSTEM, "YOU ARE THE SCOUT")


async def test_a_default_agent_gets_the_packaged_system_prompt(registry):
    agent = Agent(ScriptedLLM(), registry)

    assert agent.history[0].role is Role.SYSTEM
    assert agent.history[0].content.strip()


async def test_clearing_the_conversation_keeps_the_system_prompt(registry):
    # /clear used to empty the history outright, which quietly took the agent's
    # identity with it: every turn after it ran with no system prompt at all.
    agent = Agent(ScriptedLLM([ContentEvent("hi")]), registry, system="YOU ARE THE SCOUT")
    await drain(agent)

    agent.reset()

    assert agent.history == [Message(Role.SYSTEM, "YOU ARE THE SCOUT")]

    await drain(agent)
    assert [m.role for m in agent.history] == [Role.SYSTEM, Role.USER, Role.AGENT]
