# config.py
# settings kept between sessions, in ~/.config/harness/config.toml

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomlkit
from tomlkit.exceptions import TOMLKitError

from harness.llm import PROVIDERS
from harness.styles import SCHEMES

# Written out the first time harness saves a setting, so the file documents itself.
TEMPLATE = '''\
# harness settings. Flags on the command line win over these.
#
#   provider = "ollama"             where the model runs: ollama or groq
#   style = "graphite"              graphite, tide, ember or classic; /style sets it
#   max_iterations = 25             rounds of tool calls before a turn stops
#   deny = ["*.pem", "secrets"]     names the agent may not touch, on top of
#                                   .git and .env; globs, matched on each part of a path
#
#   [ollama]                        a table per provider, at the end of the file
#   model = "qwen3.5:9b"
#   num_ctx = 16384                 context window in tokens; bigger costs memory
#   think = false                   for models that can't think
#
#   [groq]
#   model = "openai/gpt-oss-120b"
#   max_retries = 6                 how often to wait out a rate limit and retry
'''


class ConfigError(Exception):
    '''The settings file could not be written.'''


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "harness" / "config.toml"


@dataclass
class Config:
    provider:       str | None = None
    style:          str | None = None
    max_iterations: int | None = None
    deny:           list[str] = field(default_factory=list)
    # Each provider's table, passed to its LLM's constructor; model included.
    providers:      dict[str, dict[str, Any]] = field(default_factory=dict)


# A setting's type, a test its value has to pass, and what to say when it doesn't.
Rule = tuple[type, Callable[[Any], bool], str]

WHOLE_ABOVE_ZERO: Rule = (int, lambda n: n > 0, "a whole number above 0")
NAME: Rule = (str, bool, "a name")

SETTINGS: dict[str, Rule] = {
    "provider":       (str, lambda v: v in PROVIDERS, f"one of {', '.join(PROVIDERS)}"),
    "style":          (str, lambda v: v in SCHEMES, f"one of {', '.join(SCHEMES)}"),
    "max_iterations": WHOLE_ABOVE_ZERO,
    "deny":           (list, lambda v: all(isinstance(x, str) and x and "/" not in x for x in v),
                       "a list of names or globs without slashes, like [\"*.pem\"]"),
}

PROVIDER_SETTINGS: dict[str, dict[str, Rule]] = {
    "ollama": {
        "model":   NAME,
        "num_ctx": WHOLE_ABOVE_ZERO,
        "think":   (bool, lambda v: True, "true or false"),
    },
    "groq": {
        "model":       NAME,
        "max_retries": (int, lambda n: n >= 0, "a whole number, 0 or more"),
    },
}


def check(rule: Rule, value: Any) -> bool:
    kind, test, _ = rule
    # TOML's true is a Python int too; it is never a number here.
    if isinstance(value, bool) and kind is not bool:
        return False
    return isinstance(value, kind) and test(value)


def load_config(path: Path | None = None) -> tuple[Config, list[str]]:
    '''Read the settings, and a note for anything in them that was skipped.

    A missing file means no settings. A bad setting is reported and ignored
    rather than stopping the CLI, so a typo never locks you out.
    '''

    path = path or config_path()
    config = Config()

    try:
        settings = tomlkit.parse(path.read_text()).unwrap()
    except FileNotFoundError:
        return config, []
    except (OSError, UnicodeDecodeError, TOMLKitError) as exc:
        return config, [f"couldn't read {path}: {exc}"]

    problems = []

    for key, value in settings.items():
        if key in SETTINGS:
            if check(SETTINGS[key], value):
                setattr(config, key, value)
            else:
                problems.append(f"{key} = {value!r} should be {SETTINGS[key][2]}")

        elif key in PROVIDER_SETTINGS and isinstance(value, dict):
            rules, options = PROVIDER_SETTINGS[key], {}
            for name, setting in value.items():
                if name not in rules:
                    problems.append(f"unknown setting {name!r} under [{key}]")
                elif check(rules[name], setting):
                    options[name] = setting
                else:
                    problems.append(f"{name} = {setting!r} under [{key}] should be {rules[name][2]}")
            config.providers[key] = options

        else:
            problems.append(f"unknown setting {key!r}")

    return config, problems


def save_setting(key: str, value: str, path: Path | None = None) -> None:
    '''Set one top-level setting, keeping everything else in the file as the
    user wrote it, comments included.'''

    path = path or config_path()

    try:
        try:
            document = tomlkit.parse(path.read_text())
        except FileNotFoundError:
            document = tomlkit.parse(TEMPLATE)

        document[key] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tomlkit.dumps(document))
    except (OSError, UnicodeDecodeError, TOMLKitError) as exc:
        raise ConfigError(f"couldn't save to {path}: {exc}") from None
