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
- A model to talk to, either:
  - [ollama](https://ollama.com) running locally, with a model pulled:
    `ollama pull qwen3.5:9b`
  - or a [Groq](https://console.groq.com) API key in `GROQ_API_KEY`. A `.env`
    file in the current directory, or a parent of it, is read at startup.
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
harness -p groq            # run on Groq instead, defaults to openai/gpt-oss-120b
harness -w ~/code/project  # point the agent somewhere else
```

## Settings

Defaults live in `~/.config/harness/config.toml` (under `$XDG_CONFIG_HOME` if
you set it). Flags override them. The file is created, with a commented header,
the first time `/style` saves a choice, and your own edits and comments survive.

```toml
provider = "ollama"
style = "tide"
max_iterations = 40          # rounds of tool calls before a turn stops
deny = ["*.pem", "secrets"]  # more names the agent may not touch

[ollama]
model = "qwen3.5:9b"
num_ctx = 16384              # context window; ollama's own default is small
think = false                # for models that can't think

[groq]
max_retries = 6              # wait out rate limits instead of failing
```

A setting harness doesn't recognise, or a value of the wrong kind, is reported
at startup and skipped.

## Workspace

The agent can only reach files under the workspace directory. Paths are resolved
before use and refused if they land outside it, `.git` and `.env` files are off
limits along with anything matching `deny` in your settings, and writes
go through a rename so a crash can't truncate a file. Version control is yours to
manage — the harness does not checkpoint or undo anything.

> **There is no confirmation step yet.** Inside the workspace the agent writes
> without asking, so point it at a directory you don't mind a model editing.

## How it works

`harness.core` is the agent and everything it needs to run. It imports no
terminal library, so it can be embedded, or driven by a front-end of your own.

| `harness/core/` | |
|--------|-|
| `models.py` | dataclasses for messages, tools, and the event stream |
| `llm.py`    | the provider boundary — an `LLM` protocol plus the ollama and Groq implementations |
| `workspace.py` | confined, atomic filesystem access — path resolution lives here |
| `tools.py`  | the tool registry and the built-in tools |
| `agent.py`  | the tool-calling loop |

| front-end | |
|--------|-|
| `cli.py`    | the terminal front-end |
| `panels.py` | several agents at once, one panel each |
| `styles.py` | colour schemes for the front-end — switch with `/style` |
| `config.py` | the settings file: reading it at startup, saving to it |

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

Driving the agent yourself is the same three pieces the CLI wires up — a
workspace, a provider, and the tools bound to that workspace:

```python
from harness.core import Agent, HostWorkspace, OllamaLLM, build_registry
from harness.core.models import ContentEvent

workspace = HostWorkspace("./project", deny=["*.pem"])
agent = Agent(OllamaLLM("qwen3.5:9b"), build_registry(workspace))

async for event in agent.run("what does workspace.py do?"):
    if isinstance(event, ContentEvent):
        print(event.content, end="", flush=True)
```

## Development

```sh
uv sync
uv run ruff check .
uv run pytest
uv build
```
