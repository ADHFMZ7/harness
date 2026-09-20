"""The core must not reach for a front-end.

`harness.core` is what an embedder imports, and what a second front-end builds
on. The day it imports rich, the boundary is gone and nobody finds out, because
every other test runs in a process where a front-end is already loaded. These
run in a fresh interpreter, where the import is the whole test.
"""

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
