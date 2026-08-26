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
harness              # defaults to qwen3.5:9b
harness -m llama3.2  # any model ollama has pulled
```

> **The file tools are not sandboxed.** `write_file` and `edit_file` will modify
> any path the process can reach, without confirmation, and `read_file` will pull
> any readable file into model context. Run it against directories you don't mind
> a model touching.

## How it works

| Module | |
|--------|-|
| `models.py` | dataclasses for messages, tools, and the event stream |
| `llm.py`    | the provider boundary — an `LLM` protocol plus the ollama implementation |
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
the model sees:

```python
@registry.register
async def list_files(dir_path: str = '.') -> list[str]:
    '''lists files in directory specified by path'''
    return await os.listdir(dir_path)
```

## Development

```sh
uv sync
uv run ruff check .
uv run pytest
uv build
```
