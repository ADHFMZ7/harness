# Harness

A small, readable agent harness for local LLMs: streams tokens and reasoning as
they arrive, calls tools in parallel, and feeds the results back until the model
is done.

```
› what is 5432 + 65453?

  ✻ thought for 12.2s
  ⚒ add(a=5432, b=65453)
    ↳ 70885
  The sum of 5432 and 65453 is 70885.
```

## Requirements

- Python 3.12+
- [ollama](https://ollama.com) running locally, with a model pulled:
  `ollama pull qwen3.5:9b`
- [ripgrep](https://github.com/BurntSushi/ripgrep) on `PATH`, for the `search_file` tool

## Install

```sh
uv sync
uv run harness
```

Or install the console script into an environment of your own:

```sh
uv pip install .
harness
```

`python -m harness` works too.

## Usage

```sh
harness                    # defaults to qwen3.5:9b, current directory
harness -m llama3.2        # any model ollama has pulled
harness -w ~/code/project  # point the agent somewhere else
```

The agent can only reach files under the workspace directory. Paths are resolved
before use and refused if they land outside it, `.git` is off limits, and writes
go through a rename so a crash can't truncate a file. Version control is yours to
manage — the harness does not checkpoint or undo anything.

> **There is no confirmation step yet.** Inside the workspace the agent writes
> without asking, so point it at a directory you don't mind a model editing.

## How it works

| Module | |
|--------|-|
| `models.py` | dataclasses for messages, tools, and the event stream |
| `llm.py`    | the provider boundary — an `LLM` protocol plus the ollama implementation |
| `workspace.py` | confined, atomic filesystem access — path resolution lives here |
| `tools.py`  | the tool registry and the built-in tools |
| `agent.py`  | the tool-calling loop |
| `cli.py`    | the terminal front-end |

`Agent.run()` is an async generator. It yields `ThinkingEvent`, `ContentEvent`,
`ToolCallEvent`, and `ToolResultEvent` as they happen, so a front-end can render
progress without knowing anything about the agent's internals. Tool calls in a
single batch run concurrently, and a tool that raises comes back as a
`ToolResult` with `is_error=True` rather than ending the turn — the model sees
the error and can correct itself.

Adding a tool is a decorator and a docstring; the docstring is the description
the model sees. Tools are registered inside `build_registry`, which binds them to
a single workspace, so a tool never reaches the filesystem directly:

```python
@registry.register
async def list_files(dir_path: str = '.') -> list[str]:
    '''lists files in directory specified by path'''
    return await workspace.list(dir_path)
```

## Development

```sh
uv sync
uv run ruff check .
uv run pytest
uv build
```
