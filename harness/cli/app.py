# cli/app.py
# a small terminal front-end for the agent

import argparse
import asyncio
from pathlib import Path

import groq
from dotenv import find_dotenv, load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.text import Text

from harness.cli.config import ConfigError, config_path, load_config, save_setting
from harness.cli.options import add_llm_arguments, make_llm
from harness.cli.prompt import COMMANDS, prompt_marker, prompt_session
from harness.cli.styles import DEFAULT, SCHEMES
from harness.cli.view import View, truncate
from harness.core.agent import DEFAULT_MAX_ITERATIONS, Agent
from harness.core.models import (
    ContentEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from harness.core.tools import build_registry
from harness.core.workspace import HostWorkspace, WorkspaceError


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
