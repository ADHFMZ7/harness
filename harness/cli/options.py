# cli/options.py
# turning command-line flags and the settings file into an LLM

import argparse

from harness.cli.config import Config
from harness.core.llm import LLM, PROVIDERS


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
