# cli/__init__.py
# the terminal front-end

"""The terminal front-end, and the pieces it is built from.

`app` wires a session together and owns the chat loop; `view` renders the
agent's event stream; `prompt` is the input line; `options` turns flags and the
settings file into an LLM. `panels` is a second, smaller front-end over the same
core — several agents at once, one panel each.

None of it is imported by `harness.core`, which is what keeps the agent usable
without a terminal.
"""

from harness.cli.app import main, run

__all__ = ["main", "run"]
