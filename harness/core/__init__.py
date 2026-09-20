# core/__init__.py
# the agent, and everything it needs to run, with no terminal attached

"""The harness core: an agent, the LLMs behind it, and the tools it may call.

Nothing here knows about a front-end. The agent reports what it is doing by
yielding events, and whatever drives it decides how to show them, so the same
core backs the chat CLI, the panel board, and anything embedding it.

The names below are the ones you build an agent out of. The message and event
vocabulary its stream is made of lives in `harness.core.models`.
"""

from harness.core.agent import DEFAULT_MAX_ITERATIONS, Agent, IterationLimit
from harness.core.llm import LLM, PROVIDERS, GroqLLM, OllamaLLM
from harness.core.tools import ToolRegistry, build_registry
from harness.core.workspace import (
    HostWorkspace,
    PathDenied,
    PathOutsideWorkspace,
    Workspace,
    WorkspaceError,
)

__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "PROVIDERS",
    "Agent",
    "GroqLLM",
    "HostWorkspace",
    "IterationLimit",
    "LLM",
    "OllamaLLM",
    "PathDenied",
    "PathOutsideWorkspace",
    "ToolRegistry",
    "Workspace",
    "WorkspaceError",
    "build_registry",
]
