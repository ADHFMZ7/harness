# cli.py
# a small terminal front-end for the agent

import argparse
import asyncio
import time

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.spinner import Spinner
from rich.text import Text

from harness.agent import Agent
from harness.llm import OllamaLLM
from harness.models import (
    ContentEvent,
    ThinkingEvent,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
)
from harness.tools import registry

DEFAULT_MODEL = "qwen3.5:9b"

COMMANDS = {
    "/help":  "show this message",
    "/tools": "list the tools the agent can call",
    "/clear": "forget the conversation so far",
    "/exit":  "leave (ctrl-d works too)",
}


def truncate(value, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class View:
    """Renders the agent's stream as a sequence of blocks.

    Only the block currently being streamed lives inside a `Live`; finished
    blocks are left behind in the scrollback in their final form.
    """

    def __init__(self, console: Console):
        self.console = console
        self.live: Live | None = None
        self.kind: str | None = None
        self.buffer = ""
        self.started = 0.0

    def wait(self, label: str = "thinking") -> None:
        """Show a spinner while the agent is busy between blocks."""
        self._open("wait", transient=True)
        assert self.live
        self.live.update(Padding(Spinner("dots", Text(label, style="dim")), (0, 2)))

    def append(self, kind: str, text: str) -> None:
        if self.kind != kind:
            self._open(kind)
        self.buffer += text
        assert self.live
        self.live.update(self._streaming())

    def _note(self, renderable) -> None:
        """Close the current block, then print something permanently."""
        self.end()
        self.console.print(renderable)

    def end(self) -> None:
        if self.live is None:
            return
        if self.kind != "wait":
            self.live.update(self._finished())
        self.live.stop()
        self.live = None
        self.kind = None
        self.buffer = ""

    def _open(self, kind: str, transient: bool = False) -> None:
        self.end()
        self.kind = kind
        self.buffer = ""
        self.started = time.monotonic()
        self.live = Live(
            console=self.console,
            refresh_per_second=15,
            transient=transient,
            vertical_overflow="visible",
        )
        self.live.start()

    def _streaming(self):
        if self.kind == "thinking":
            tail = self.buffer[-(self.console.width * 3) :].strip()
            return Padding(Text(tail, style="dim italic"), (0, 2))
        return Padding(Text(self.buffer.lstrip()), (0, 2))

    def _finished(self):
        if self.kind == "thinking":
            elapsed = time.monotonic() - self.started
            return Padding(Text(f"✻ thought for {elapsed:.1f}s", style="dim"), (0, 2))
        body = self.buffer.strip()
        return Padding(Markdown(body) if body else Text(""), (0, 2))

    def tool_call(self, call: ToolCall) -> None:
        args = ", ".join(f"{k}={truncate(v, 40)}" for k, v in call.arguments.items())
        line = Text("  ⚒ ", style="magenta")
        line.append(call.name, style="bold magenta")
        line.append(f"({args})", style="dim")
        self._note(line)

    def tool_result(self, result: ToolResult) -> None:
        body = truncate(result.result, self.console.width - 8)
        style = "red" if result.is_error else "dim"
        self._note(Text(f"    ↳ {body}", style=style))


async def turn(agent: Agent, view: View, console: Console, prompt: str) -> None:
    problem = None

    console.print()
    view.wait()
    try:
        async for event in agent.run(prompt):
            match event:
                case ThinkingEvent():
                    view.append("thinking", event.thinking)
                case ContentEvent():
                    view.append("content", event.content)
                case ToolCallEvent():
                    for call in event.tool_calls:
                        view.tool_call(call)
                    view.wait("running")
                case ToolResultEvent():
                    for result in event.results:
                        view.tool_result(result)
                    view.wait()
    except KeyboardInterrupt:
        problem = "interrupted"
    except Exception as exc:
        problem = f"{type(exc).__name__}: {exc}"
    finally:
        view.end()

    if problem:
        console.print(Text(f"  ⨯ {problem}", style="red"))
    console.print()


def command(console: Console, agent: Agent, line: str) -> bool:
    """Run a slash command. Returns True when it's time to quit."""
    name = line.split()[0].lower()

    match name:
        case "/exit" | "/quit":
            return True

        case "/clear":
            agent.history.clear()
            console.print("  [dim]conversation cleared[/]\n")

        case "/tools":
            console.print()
            for tool in registry.get_tools():
                summary = truncate(tool.description, console.width - 24)
                console.print(f"  [bold]{tool.name:<14}[/][dim]{summary}[/]")
            console.print()

        case "/help":
            console.print()
            for cmd, summary in COMMANDS.items():
                console.print(f"  [bold]{cmd:<14}[/][dim]{summary}[/]")
            console.print()

        case _:
            console.print(f"  [red]unknown command[/] [dim]{name}[/] — try /help\n")

    return False


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog="harness", description="chat with a tool-using agent"
    )
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, 
        help=f"ollama model (default: {DEFAULT_MODEL})"
    )
    args = parser.parse_args()

    console = Console()
    agent = Agent(OllamaLLM(args.model), registry)
    view = View(console)

    console.print()
    console.print(f"  [bold cyan]harness[/] [dim]· {args.model}[/]")
    console.print("  [dim]/help for commands · ctrl-d to exit[/]")
    console.print()

    while True:
        try:
            prompt = console.input("[bold cyan]›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]bye[/]\n")
            return

        if not prompt:
            continue

        if prompt.startswith("/"):
            if command(console, agent, prompt):
                console.print("  [dim]bye[/]\n")
                return
            continue

        await turn(agent, view, console, prompt)


def run() -> None:
    """Console-script entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
