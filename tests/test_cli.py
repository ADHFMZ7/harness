import asyncio
import io
import re
from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.default_styles import DEFAULT_STYLES
from rich.padding import Padding
from rich.style import Style

from harness import cli
from harness.agent import Agent
from harness.cli import (
    COMMANDS,
    AgentMarkdown,
    CommandCompleter,
    View,
    command,
    prompt_session,
    settled,
)
from harness.config import load_config
from harness.styles import DEFAULT, SCHEMES
from harness.tools import ToolRegistry


def completions(text: str) -> list[str]:
    found = CommandCompleter().get_completions(Document(text), CompleteEvent())
    return [completion.text for completion in found]


def test_a_slash_offers_every_command():
    assert completions("/") == list(COMMANDS)


def test_a_prefix_narrows_the_commands():
    assert completions("/c") == ["/clear"]
    assert completions("/T") == ["/tools"]


def test_prompts_that_are_not_commands_are_left_alone():
    assert completions("summarize /help") == []
    assert completions("/help me") == []


async def until(condition) -> None:
    for _ in range(200):
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("the prompt never got there")


async def test_tab_cycles_through_the_commands():
    with create_pipe_input() as pipe:
        session = prompt_session(input=pipe, output=DummyOutput())
        buffer = session.default_buffer
        answer = asyncio.create_task(session.prompt_async())

        # One key at a time, the way a person types: the menu fills in the
        # background, so keys sent all at once would outrun it.
        pipe.send_text("/")
        await until(lambda: buffer.complete_state is not None)
        pipe.send_text("\t")
        await until(lambda: buffer.text == "/help")
        pipe.send_text("\t")
        await until(lambda: buffer.text == "/tools")
        pipe.send_text("\x1b[Z")  # shift-tab
        await until(lambda: buffer.text == "/help")
        pipe.send_text("\r")

        assert await answer == "/help"


def test_tables_have_rules_between_columns_and_rows():
    console = Console(file=io.StringIO(), width=40, color_system=None, record=True)
    console.print(AgentMarkdown("| a | b |\n|---|---|\n| one | two |\n| three | four |"))

    lines = [line.strip() for line in console.export_text().splitlines() if line.strip()]

    assert lines[0].replace(" ", "") == "a│b"
    assert "┿" in lines[1]                        # heavy rule under the header
    assert lines[2].replace(" ", "") == "one│two"
    assert "┼" in lines[3]                        # light rule between rows
    assert lines[4].replace(" ", "") == "three│four"


def test_only_blocks_before_the_last_one_are_final():
    text = "# Title\n\nFirst paragraph.\n\nSecond, still being wri"
    final, _ = settled(text)
    assert text[:final] == "# Title\n\nFirst paragraph.\n\n"


def test_an_open_code_fence_is_never_split():
    text = "Intro.\n\n```python\nx = 1\n\ny = 2\n"
    final, _ = settled(text)
    assert text[:final] == "Intro.\n\n"


def test_a_list_stays_open_while_it_can_gain_items():
    assert settled("- one\n\n- two\n") == (0, False)


DOCUMENT = """## Summary

Streaming **markdown** with `inline code` and a [link](https://example.com).

- first item
- second item that is long enough that it has to wrap onto another line

| Module | Purpose |
|--------|---------|
| `cli.py` | the terminal front-end |
| `llm.py` | the provider boundary |

```python
def add(a, b):

    return a + b
```

---
1. after a rule
2. numbered

> a quote to finish
"""


def test_streaming_prints_what_rendering_it_whole_would():
    streamed = Console(file=io.StringIO(), width=60, color_system=None)
    view = View(streamed)
    for start in range(0, len(DOCUMENT), 7):
        view.append("content", DOCUMENT[start : start + 7])
    view.end()

    whole = Console(file=io.StringIO(), width=60, color_system=None)
    whole.print(Padding(AgentMarkdown(DOCUMENT), (0, 2)))

    def visible(console: Console) -> list[str]:
        return [line.rstrip() for line in console.file.getvalue().splitlines()]

    assert visible(streamed) == visible(whole)


def test_style_completes_scheme_names():
    assert completions("/sty") == ["/style"]
    assert completions("/style ") == list(SCHEMES)
    assert completions("/style t") == ["tide"]
    assert completions("/style tide x") == []


def quiet_view() -> View:
    return View(Console(file=io.StringIO(), width=60))


def test_style_alone_cycles_to_the_next_scheme():
    view = quiet_view()
    names = list(SCHEMES)
    assert view.scheme is DEFAULT

    for expected in names[1:] + names[:1]:
        command(view.console, Agent(None, ToolRegistry()), view, "/style")
        assert view.scheme.name == expected


def test_style_picks_a_scheme_by_name_and_ignores_unknown_ones():
    view = quiet_view()
    agent = Agent(None, ToolRegistry())

    command(view.console, agent, view, "/style Ember")
    assert view.scheme.name == "ember"

    command(view.console, agent, view, "/style neon")
    assert view.scheme.name == "ember"
    assert "no style called neon" in view.console.file.getvalue()


def test_switching_schemes_replaces_the_old_styles():
    view = quiet_view()
    view.use(SCHEMES["tide"])
    assert view.console.get_style("harness.accent") == Style.parse("bold #88c0d0")

    # classic sets no markdown styles, so tide's must not linger underneath
    view.use(SCHEMES["classic"])
    assert view.console.get_style("markdown.h2") == Style.parse(str(DEFAULT_STYLES["markdown.h2"]))


def test_every_scheme_defines_every_style_the_cli_uses():
    used = set(re.findall(r'"(harness\.[a-z.]+)"', Path(cli.__file__).read_text()))
    used |= set(re.findall(r"\[(harness\.[a-z.]+)\]", Path(cli.__file__).read_text()))

    for scheme in SCHEMES.values():
        assert used <= scheme.styles.keys(), scheme.name


def test_the_chosen_style_is_saved_for_next_time():
    view = quiet_view()

    command(view.console, Agent(None, ToolRegistry()), view, "/style tide")

    assert load_config()[0].style == "tide"
