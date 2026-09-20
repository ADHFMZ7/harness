# cli.py
# a small terminal front-end for the agent

import argparse
import asyncio
import time
from collections.abc import Iterable
from pathlib import Path

import groq
from dotenv import find_dotenv, load_dotenv
from markdown_it import MarkdownIt
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI
from rich import box
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import Markdown, TableElement
from rich.markup import escape
from rich.padding import Padding
from rich.segment import SegmentLines
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from harness.config import Config, ConfigError, config_path, load_config, save_setting
from harness.core.agent import DEFAULT_MAX_ITERATIONS, Agent
from harness.core.llm import LLM, PROVIDERS
from harness.core.models import (
    ContentEvent,
    ThinkingEvent,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
)
from harness.core.tools import build_registry
from harness.core.workspace import HostWorkspace, WorkspaceError
from harness.styles import DEFAULT, SCHEMES, Scheme

COMMANDS = {
    "/help":  "show this message",
    "/tools": "list the tools the agent can call",
    "/clear": "forget the conversation so far",
    "/style": "switch colours: alone to cycle, or /style <name>",
    "/exit":  "leave (ctrl-d works too)",
}

# The parser rich's Markdown uses, so blocks split the way rich renders them.
MARKDOWN = MarkdownIt().enable("strikethrough").enable("table")
OPENS_WITH_BLANK = {"bullet_list_open", "ordered_list_open", "blockquote_open", "table_open"}


def truncate(value, limit: int) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class DividedTable(TableElement):
    """A markdown table with a rule between every column and every row.

    Rich's default leaves one space between cells, so a cell that wraps runs
    into its neighbours and its second line reads like a new row.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        for table in super().__rich_console__(console, options):
            if isinstance(table, Table):
                table.box = box.MINIMAL_HEAVY_HEAD
                table.show_edge = False
                table.show_lines = True
                table.collapse_padding = False
                table.pad_edge = True
            yield table


class AgentMarkdown(Markdown):
    """Markdown as the agent's answers are shown."""

    elements = {**Markdown.elements, "table_open": DividedTable}


class CommandCompleter(Completer):
    """Completes slash commands, and the scheme names after /style, with a
    summary of each alongside."""

    def get_completions(self, document: Document,
                        complete_event: CompleteEvent) -> Iterable[Completion]:
        text = document.text_before_cursor.lower()
        if not text.startswith("/"):
            return

        command, space, argument = text.partition(" ")
        if not space:
            choices = COMMANDS
        elif command == "/style" and " " not in argument:
            choices = {name: scheme.summary for name, scheme in SCHEMES.items()}
        else:
            return

        word = argument if space else text
        for name, summary in choices.items():
            if name.startswith(word):
                yield Completion(name, start_position=-len(word), display_meta=summary)


def prompt_marker(console: Console) -> ANSI:
    """The › before the input line, in the current scheme."""
    with console.capture() as capture:
        console.print(Text("›", style="harness.accent"), end=" ")
    return ANSI(capture.get())


def prompt_session(**io) -> PromptSession:
    """The input line: the menu opens as a command is typed, tab and shift-tab
    cycle through it, and the up arrow recalls earlier prompts."""
    return PromptSession(completer=CommandCompleter(), complete_while_typing=True, **io)


def settled(text: str) -> tuple[int, bool]:
    """How much of a markdown text still being written is final, and whether
    that part ends in a horizontal rule.

    Only the last top-level block can still change: a paragraph can turn into
    a heading or a table, a list can gain items, a fence stays open until it
    closes. Everything before the last block is final.
    """
    blocks = [
        (token.type, token.map[0])
        for token in MARKDOWN.parse(text)
        if token.level == 0 and token.nesting >= 0 and token.map
    ]
    if len(blocks) < 2:
        return 0, False

    offset = 0
    for _ in range(blocks[-1][1]):
        offset = text.index("\n", offset) + 1
    return offset, blocks[-2][0] == "hr"


class Tail:
    """As many of a renderable's last lines as fit on the screen.

    A live region taller than the screen can't be redrawn in place: the lines
    that scroll off the top are left behind, half written, in the scrollback.
    """

    def __init__(self, renderable):
        self.renderable = renderable

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        lines = console.render_lines(self.renderable, options.update(height=None), pad=False)
        yield SegmentLines(lines[-max(1, console.height - 2) :], new_lines=True)


class View:
    """Renders the agent's stream as a sequence of blocks.

    The block being streamed lives in a transient `Live`. An answer is printed
    for good one markdown block at a time, as soon as each block is final, so
    only the block still being written is live. The scrollback only ever holds
    finished, rendered output.
    """

    def __init__(self, console: Console, scheme: Scheme = DEFAULT):
        self.console = console
        self.scheme: Scheme | None = None
        self.use(scheme)
        self.live: Live | None = None
        self.kind: str | None = None
        self.buffer = ""
        self.printed = 0      # how much of the buffer is already in the scrollback
        self.gap = False      # whether the next markdown printed needs a blank line first
        self.started = 0.0

    def use(self, scheme: Scheme) -> None:
        """Switch colour scheme. What is already in the scrollback keeps its colours."""
        if self.scheme is not None:
            self.console.pop_theme()
        self.console.push_theme(scheme.theme)
        self.scheme = scheme

    def wait(self, label: str = "thinking") -> None:
        """Show a spinner while the agent is busy between blocks."""
        self._open("wait")
        assert self.live
        self.live.update(Padding(Spinner("dots", Text(label, style="harness.muted")), (0, 2)))

    def append(self, kind: str, text: str) -> None:
        if self.kind != kind:
            self._open(kind)
        self.buffer += text
        if kind == "content":
            final, rule = settled(self.buffer[self.printed :])
            if final:
                self._print_markdown(self.buffer[self.printed : self.printed + final])
                self.printed += final
                self.gap = not rule
        assert self.live
        self.live.update(self._streaming())

    def _note(self, renderable) -> None:
        """Close the current block, then print something permanently."""
        self.end()
        self.console.print(renderable)

    def end(self) -> None:
        if self.live is None:
            return
        self.live.stop()
        if self.kind == "thinking":
            elapsed = time.monotonic() - self.started
            self.console.print(Padding(Text(f"✻ thought for {elapsed:.1f}s", style="harness.muted"), (0, 2)))
        elif self.kind == "content":
            self._print_markdown(self.buffer[self.printed :])
        self.live = None
        self.kind = None

    def _open(self, kind: str) -> None:
        self.end()
        self.kind = kind
        self.buffer = ""
        self.printed = 0
        self.gap = False
        self.started = time.monotonic()
        self.live = Live(console=self.console, refresh_per_second=15, transient=True)
        self.live.start()

    def _print_markdown(self, text: str) -> None:
        tokens = MARKDOWN.parse(text)
        if not tokens:
            return
        # Rich opens a list, table or quote with a blank line of its own, so
        # only add one before other blocks.
        if self.gap and tokens[0].type not in OPENS_WITH_BLANK:
            self.console.print()
        self.console.print(Padding(self._markdown(text), (0, 2)))

    def _streaming(self):
        if self.kind == "thinking":
            tail = self.buffer[-(self.console.width * 3) :].strip()
            return Tail(Padding(Text(tail, style="harness.thinking"), (0, 2)))
        return Tail(Padding(self._markdown(self.buffer[self.printed :]), (0, 2)))

    def _markdown(self, text: str) -> AgentMarkdown:
        assert self.scheme
        return AgentMarkdown(text, code_theme=self.scheme.code_theme)

    def tool_call(self, call: ToolCall) -> None:
        args = ", ".join(f"{k}={truncate(v, 40)}" for k, v in call.arguments.items())
        self._note(Text.assemble(
            ("  ⚒ ", "harness.tool"),
            (call.name, "harness.tool.name"),
            (f"({args})", "harness.muted"),
        ))

    def tool_result(self, result: ToolResult) -> None:
        body = truncate(result.result, self.console.width - 8)
        style = "harness.error" if result.is_error else "harness.muted"
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
        console.print(Text(f"  ⨯ {problem}", style="harness.error"))
    console.print()


def command(console: Console, agent: Agent, view: View, line: str) -> bool:
    """Run a slash command. Returns True when it's time to quit."""
    name, *arguments = line.lower().split()

    match name:
        case "/exit" | "/quit":
            return True

        case "/clear":
            agent.history.clear()
            console.print("  [harness.muted]conversation cleared[/]\n")

        case "/tools":
            console.print()
            for tool in agent.tools.get_tools():
                summary = truncate(tool.description, console.width - 24)
                console.print(f"  [bold]{tool.name:<14}[/][harness.muted]{summary}[/]")
            console.print()

        case "/help":
            console.print()
            for cmd, summary in COMMANDS.items():
                console.print(f"  [bold]{cmd:<14}[/][harness.muted]{summary}[/]")
            console.print(f"\n  [harness.muted]settings · {escape(home_relative(config_path()))}[/]")
            console.print()

        case "/style":
            style(console, view, arguments)

        case _:
            console.print(f"  [harness.error]unknown command[/] "
                          f"[harness.muted]{escape(name)}[/] — try /help\n")

    return False


def style(console: Console, view: View, arguments: list[str]) -> None:
    """Switch to the named scheme, or with no name, the next one in turn."""
    assert view.scheme
    names = list(SCHEMES)

    if not arguments:
        name = names[(names.index(view.scheme.name) + 1) % len(names)]
    elif arguments[0] in SCHEMES:
        name = arguments[0]
    else:
        console.print(f"  [harness.error]no style called[/] [harness.muted]{escape(arguments[0])}[/]"
                      f" — try {', '.join(names)}\n")
        return

    view.use(SCHEMES[name])
    console.print(Text.assemble(
        ("  style ", "harness.muted"),
        (name, "harness.accent"),
        (f" · {SCHEMES[name].summary}", "harness.muted"),
    ))
    try:
        save_setting("style", name)
    except ConfigError as exc:
        console.print(f"  [harness.error]not saved:[/] [harness.muted]{escape(str(exc))}[/]")
    console.print(Text.assemble(
        ("  ⚒ ", "harness.tool"),
        ("read_file", "harness.tool.name"),
        ("(path=README.md)", "harness.muted"),
    ))
    console.print()


def add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = ", ".join(f"{model} on {name}" for name, (_, model) in PROVIDERS.items())
    parser.add_argument(
        "-p", "--provider", choices=PROVIDERS,
        help="where the model runs (default: ollama, or provider in the settings file)"
    )
    parser.add_argument(
        "-m", "--model",
        help=f"model name (default: {defaults})"
    )


def make_llm(args: argparse.Namespace, config: Config) -> tuple[LLM, str]:
    """The LLM to use, and the name of its model. A flag beats the settings
    file, which beats the built-in default."""
    provider = args.provider or config.provider or "ollama"
    make, default_model = PROVIDERS[provider]
    options = dict(config.providers.get(provider, {}))
    saved_model = options.pop("model", None)
    model = args.model or saved_model or default_model
    return make(model, **options), model


def home_relative(path: Path) -> str:
    home = Path.home()
    return f"~/{path.relative_to(home)}" if path.is_relative_to(home) else str(path)


async def main() -> None:
    parser = argparse.ArgumentParser(
        prog="harness", description="chat with a tool-using agent"
    )
    add_llm_arguments(parser)
    parser.add_argument(
        "-w", "--workspace", default=".",
        help="directory the agent may read and write (default: the current one)"
    )
    args = parser.parse_args()

    config, problems = load_config()
    console = Console()
    view = View(console, SCHEMES.get(config.style or "", DEFAULT))

    for problem in problems:
        console.print(f"  [harness.error]settings:[/] [harness.muted]{escape(problem)}[/]")

    try:
        workspace = HostWorkspace(args.workspace, deny=config.deny)
        llm, model = make_llm(args, config)
    except (WorkspaceError, groq.GroqError) as exc:
        console.print(f"\n  [harness.error]{escape(str(exc))}[/]\n")
        return

    agent = Agent(llm, build_registry(workspace),
                  max_iterations=config.max_iterations or DEFAULT_MAX_ITERATIONS)
    session = prompt_session()

    console.print()
    console.print(f"  [harness.accent]harness[/] [harness.muted]· {model}[/]")
    console.print(f"  [harness.muted]workspace · {workspace.root}[/]")
    console.print("  [harness.muted]/help for commands · ctrl-d to exit[/]")
    console.print()

    while True:
        try:
            prompt = (await session.prompt_async(prompt_marker(console))).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [harness.muted]bye[/]\n")
            return

        if not prompt:
            continue

        if prompt.startswith("/"):
            if command(console, agent, view, prompt):
                console.print("  [harness.muted]bye[/]\n")
                return
            continue

        await turn(agent, view, console, prompt)


def run() -> None:
    """Console-script entry point."""
    # API keys, e.g. GROQ_API_KEY, can live in a .env file.
    load_dotenv(find_dotenv(usecwd=True))
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
