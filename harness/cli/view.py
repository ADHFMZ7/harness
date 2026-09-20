# cli/view.py
# renders the agent's event stream into the terminal

import time

from markdown_it import MarkdownIt
from rich import box
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import Markdown, TableElement
from rich.padding import Padding
from rich.segment import SegmentLines
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from harness.cli.styles import DEFAULT, Scheme
from harness.core.models import ToolCall, ToolResult

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
