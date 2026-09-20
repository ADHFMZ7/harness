"""The core must not reach for a front-end.

`harness.core` is what an embedder imports, and what a second front-end builds
on. The day it imports rich, the boundary is gone and nobody finds out, because
every other test runs in a process where a front-end is already loaded. These
run in a fresh interpreter, where the import is the whole test.
"""

import pathlib
import re
import subprocess
import sys

FRONT_END = ("rich", "prompt_toolkit", "markdown_it", "dotenv")


def run(source: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=60
    )


def test_core_pulls_in_no_front_end():
    result = run(
        "import sys, harness.core\n"
        f"front_end = {FRONT_END!r}\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in front_end)\n"
        "print(','.join(leaked))\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"harness.core imported {result.stdout.strip()}"


def test_core_submodules_pull_in_no_front_end():
    # Imported one at a time, so a stray import is pinned to its module rather
    # than hidden behind whichever sibling the package happens to load first.
    for name in ("agent", "llm", "models", "tools", "workspace"):
        result = run(
            f"import sys, harness.core.{name}\n"
            f"front_end = {FRONT_END!r}\n"
            "leaked = sorted(m for m in sys.modules if m.split('.')[0] in front_end)\n"
            "print(','.join(leaked))\n"
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "", (
            f"harness.core.{name} imported {result.stdout.strip()}"
        )


def test_public_surface_is_importable():
    result = run(
        "from harness.core import (\n"
        "    Agent, GroqLLM, HostWorkspace, OllamaLLM, ToolRegistry,\n"
        "    Workspace, WorkspaceError, build_registry,\n"
        ")\n"
        "from harness.core.models import ContentEvent, ThinkingEvent, ToolCallEvent\n"
    )

    assert result.returncode == 0, result.stderr


def test_the_front_end_names_no_provider_sdk():
    """A front-end should not have to know which provider it built.

    Catching a provider's own error type means importing its SDK, and then
    every new provider edits every front-end. `ProviderError` is what the core
    promises instead, so nothing under `harness/cli/` mentions an SDK by name.
    """
    import harness.cli

    package = pathlib.Path(harness.cli.__file__).parent
    offenders = {
        path.name: [
            line.strip()
            for line in path.read_text().splitlines()
            if re.match(r"^\s*(import|from)\s+(groq|ollama)\b", line)
        ]
        for path in sorted(package.glob("*.py"))
    }

    assert not {name: found for name, found in offenders.items() if found}
