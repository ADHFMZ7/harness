# panels.py
# runs several agents at once, one panel each

import argparse
import asyncio
import math
import time
from dataclasses import dataclass

import groq
from dotenv import find_dotenv, load_dotenv
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from harness.cli import AgentMarkdown, add_llm_arguments, make_llm, truncate
from harness.config import load_config
from harness.core.agent import DEFAULT_MAX_ITERATIONS, Agent
from harness.core.models import (
    ContentEvent,
    ThinkingEvent,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
)
from harness.core.tools import ToolRegistry, build_registry
from harness.core.workspace import HostWorkspace, WorkspaceError

DEFAULT_TASKS = [
    "List the files in the workspace and say in two sentences what this project is.",
    "Read README.md and summarize it in three bullet points.",
    "Search the workspace for the word 'async' and tell me which files use it most.",
    "Use the add tool to work out 1234 + 5678 + 91011, one addition at a time.",
]

# Agents sharing a workspace can lose each other's edits, so they only get
# tools that read.
READ_ONLY_TOOLS = ("add", "list_files", "read_file", "search_file")

FRAME   = 0.1
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


@dataclass
class Block:
    kind:    str
    text:    str
    started: float
    ended:   float | None = None

    def render(self, limit: int) -> Text:
        match self.kind:
            case "thinking" if self.ended is not None:
                return Text(f"✻ thought for {self.ended - self.started:.1f}s", style="dim")
            case "thinking":
                return Text(self.text[-limit:].strip(), style="dim italic")
            case "content":
                return Text(self.text[-limit:].strip())
            case "call":
                return Text(self.text, style="magenta")
            case "error":
                return Text(self.text, style="red")
            case _:
                return Text(self.text, style="dim")


class Pane:
    """One agent's panel, built up from the events it yields."""

    def __init__(self, number: int, task: str):
        self.number = number
        self.task   = task
        self.blocks: list[Block] = []
        self.status = "waiting"
        self.calls  = 0

        self.started = time.monotonic()
        self.first_token: float | None = None
        self.finished:    float | None = None

    @property
    def answer(self) -> str:
        for block in reversed(self.blocks):
            if block.kind == "content":
                return block.text.strip()
        return ""

    def stream(self, kind: str, text: str) -> None:
        if self.first_token is None:
            self.first_token = time.monotonic()
        if not self.blocks or self.blocks[-1].kind != kind:
            self._open(kind)
        self.blocks[-1].text += text
        self.status = "thinking" if kind == "thinking" else "writing"

    def tool_call(self, call: ToolCall) -> None:
        args = ", ".join(f"{k}={truncate(v, 30)}" for k, v in call.arguments.items())
        self._open("call", f"⚒ {call.name}({args})")
        self.calls += 1
        self.status = "running tools"

    def tool_result(self, result: ToolResult) -> None:
        self._open("error" if result.is_error else "result", f"↳ {truncate(result.result, 120)}")
        self.status = "waiting"

    def finish(self, problem: str | None = None) -> None:
        if problem:
            self._open("error", f"⨯ {problem}")
        self._close()
        self.finished = time.monotonic()
        self.status = "failed" if problem else "done"

    def _open(self, kind: str, text: str = "") -> None:
        self._close()
        self.blocks.append(Block(kind, text, time.monotonic()))

    def _close(self) -> None:
        if self.blocks and self.blocks[-1].ended is None:
            self.blocks[-1].ended = time.monotonic()

    def render(self, console: Console, width: int, height: int) -> Panel:
        inner = width - 5  # borders, padding, and a column of slack
        lines = height - 2

        # Walk back from the newest block until the panel is full, so a long
        # history costs nothing to draw.
        shown: list[Text] = []
        for block in reversed(self.blocks):
            text = block.render(inner * lines)
            if text.plain:
                shown[:0] = text.wrap(console, inner)
            if len(shown) >= lines:
                break

        now = self.finished or time.monotonic()
        if self.finished:
            border = "red" if self.status == "failed" else "green"
            status = self.status
        else:
            border = "dim" if self.status == "waiting" else "cyan"
            status = f"{SPINNER[int(now * 10) % len(SPINNER)]} {self.status}"

        return Panel(
            Text("\n").join(shown[-lines:]),
            title=Text(f" {self.number} · {truncate(self.task, inner - 8)} "),
            title_align="left",
            subtitle=Text(f" {status} · {now - self.started:.1f}s "),
            subtitle_align="right",
            border_style=border,
            height=height,
            padding=(0, 1),
        )


class Board:
    """Lays the panes out in a grid and redraws them on a timer."""

    def __init__(self, console: Console, panes: list[Pane], model: str):
        self.console = console
        self.panes   = panes
        self.model   = model
        self.started = time.monotonic()
        self.lag     = 0.0

    async def animate(self, live: Live) -> None:
        # This runs on the same loop as the agents. If anything blocks it, the
        # screen freezes and the stall shows up here as lag.
        loop = asyncio.get_running_loop()
        while True:
            before = loop.time()
            await asyncio.sleep(FRAME)
            self.lag = max(self.lag, loop.time() - before - FRAME)
            live.update(self.render(), refresh=True)

    def render(self) -> Group:
        cols = math.ceil(math.sqrt(len(self.panes)))
        rows = math.ceil(len(self.panes) / cols)

        width  = self.console.width // cols
        height = max(6, (self.console.height - 2) // rows)

        grid = Table.grid(expand=True)
        for _ in range(cols):
            grid.add_column(ratio=1)
        for row in range(rows):
            panes = self.panes[row * cols : (row + 1) * cols]
            grid.add_row(*(pane.render(self.console, width, height) for pane in panes))

        return Group(self.header(), grid)

    def header(self) -> Text:
        done = sum(pane.finished is not None for pane in self.panes)
        elapsed = time.monotonic() - self.started

        text = Text("  harness", style="bold cyan")
        text.append(
            f" · {self.model} · {done}/{len(self.panes)} done · {elapsed:.0f}s"
            f" · loop lag max {self.lag * 1000:.0f}ms",
            style="dim",
        )
        return text


def read_only(registry: ToolRegistry) -> ToolRegistry:
    subset = ToolRegistry()
    for name in READ_ONLY_TOOLS:
        subset.register(registry[name].function)
    return subset


async def drive(agent: Agent, pane: Pane) -> None:
    try:
        async for event in agent.run(pane.task):
            match event:
                case ThinkingEvent():
                    pane.stream("thinking", event.thinking)
                case ContentEvent():
                    pane.stream("content", event.content)
                case ToolCallEvent():
                    for call in event.tool_calls:
                        pane.tool_call(call)
                case ToolResultEvent():
                    for result in event.results:
                        pane.tool_result(result)
    except Exception as exc:
        pane.finish(f"{type(exc).__name__}: {exc}")
    else:
        pane.finish()


def report(console: Console, panes: list[Pane]) -> None:
    def since_start(pane: Pane, moment: float | None) -> str:
        return "—" if moment is None else f"{moment - pane.started:.1f}s"

    table = Table("", "first token", "finished", "tool calls", "", box=None,
                  header_style="dim", padding=(0, 2))
    for pane in panes:
        table.add_row(str(pane.number), since_start(pane, pane.first_token),
                      since_start(pane, pane.finished), str(pane.calls), pane.status)

    console.print(Padding(table, (1, 0)))

    for pane in panes:
        console.print(Text(f"  {pane.number} · {pane.task}", style="bold"))
        body = AgentMarkdown(pane.answer) if pane.answer else Text("no answer", style="dim")
        console.print(Padding(body, (0, 4)))
        console.print()


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog="harness.panels", description="run several agents at once, one panel each"
    )
    parser.add_argument(
        "tasks", nargs="*",
        help="one prompt per agent (default: four read-only demo tasks)"
    )
    add_llm_arguments(parser)
    parser.add_argument(
        "-w", "--workspace", default=".",
        help="directory the agents may read (default: the current one)"
    )
    args = parser.parse_args()

    config, problems = load_config()
    console = Console()

    for problem in problems:
        console.print(f"  [red]settings:[/] [dim]{escape(problem)}[/]")

    try:
        workspace = HostWorkspace(args.workspace, deny=config.deny)
        # One client for every agent; each agent keeps its own history.
        llm, model = make_llm(args, config)
    except (WorkspaceError, groq.GroqError) as exc:
        console.print(f"\n  [red]{exc}[/]\n")
        return

    tools = read_only(build_registry(workspace))
    panes = [Pane(number, task) for number, task in enumerate(args.tasks or DEFAULT_TASKS, 1)]
    board = Board(console, panes, model)

    with Live(board.render(), console=console, auto_refresh=False) as live:
        redraw = asyncio.create_task(board.animate(live))
        try:
            rounds = config.max_iterations or DEFAULT_MAX_ITERATIONS
            agents = [Agent(llm, tools, max_iterations=rounds) for _ in panes]
            await asyncio.gather(*(drive(agent, pane) for agent, pane in zip(agents, panes, strict=True)))
        finally:
            redraw.cancel()
            live.update(board.render(), refresh=True)

    report(console, panes)


def run() -> None:
    load_dotenv(find_dotenv(usecwd=True))
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
