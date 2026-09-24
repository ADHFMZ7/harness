# cli/prompt.py
# the input line: slash commands, completion and the prompt marker

from collections.abc import Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI
from rich.console import Console
from rich.text import Text

from harness.cli.styles import SCHEMES

COMMANDS = {
    "/help":  "show this message",
    "/tools": "list the tools the agent can call",
    "/clear": "forget the conversation so far",
    "/style": "switch colours: alone to cycle, or /style <name>",
    "/exit":  "leave (ctrl-d works too)",
}


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
