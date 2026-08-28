# workspace.py
# confines the agent's file access to a single directory

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Protocol

import aiofiles

MAX_READ_BYTES = 2 * 1024 * 1024
NEW_FILE_MODE  = 0o644

# Writing into .git can corrupt the user's history, and hooks there execute.
DENIED_NAMES = frozenset({".git"})


# These reach the model as "ClassName: message", so both halves are interface.

class WorkspaceError(Exception):
    '''An operation the agent asked for cannot be carried out.'''

class PathOutsideWorkspace(WorkspaceError):
    '''The path resolved somewhere the agent is not allowed to reach.'''

class PathDenied(WorkspaceError):
    '''The path is inside the workspace but off limits anyway.'''


class Workspace(Protocol):
    root: Path

    async def read(self, path: str, start_line: int | None = None,
                   end_line: int | None = None) -> str: ...
    async def write(self, path: str, content: str) -> None: ...
    async def edit(self, path: str, old: str, new: str) -> None: ...
    async def list(self, path: str = '.') -> list[str]: ...
    async def search(self, query: str, path: str = '.', max_results: int = 50) -> str: ...


class HostWorkspace(Workspace):

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

        if not self.root.is_dir():
            raise WorkspaceError(f"Workspace root {self.root} is not a directory.")


    def resolve(self, path: str) -> Path:
        '''Turn a path the agent gave us into an absolute path inside the root.'''

        if not path or not path.strip():
            raise WorkspaceError("No path given. Paths are relative to the workspace root.")

        if "\x00" in path:
            raise WorkspaceError("Paths cannot contain null bytes.")

        candidate = Path(path)

        # parts is empty for ".", so the guard has to come first
        if candidate.is_absolute() or (candidate.parts and candidate.parts[0].startswith("~")):
            raise PathOutsideWorkspace(
                f"{path!r} is not relative to the workspace root ({self.root})."
            )

        # resolve() collapses '..' and symlinks together, so a link inside the
        # workspace pointing out is caught here too, whether it is the last
        # component or a directory on the way.
        target = (self.root / candidate).resolve()

        if not target.is_relative_to(self.root):
            raise PathOutsideWorkspace(
                f"{path!r} resolves outside the workspace root ({self.root})."
            )

        for part in target.relative_to(self.root).parts:
            if part in DENIED_NAMES:
                raise PathDenied(f"{path!r} is inside {part}, which is off limits.")

        return target


    async def read(self, path: str, start_line: int | None = None,
                   end_line: int | None = None) -> str:
        '''Read a file, optionally bounded by an inclusive line range.'''

        target = self.resolve(path)

        lines = []
        size  = 0
        line_num = 0

        try:
            async with aiofiles.open(target, "rb") as file:
                async for line in file:
                    line_num += 1
                    if start_line is not None and line_num < start_line:
                        continue
                    if end_line is not None and line_num > end_line:
                        break

                    size += len(line)
                    if size > MAX_READ_BYTES:
                        raise WorkspaceError(
                            f"Reading {path!r} passed the {MAX_READ_BYTES} byte limit. "
                            "Narrow it down with start_line and end_line."
                        )

                    lines.append(self._decode(line, path))
        except FileNotFoundError:
            raise WorkspaceError(f"{path!r} does not exist.") from None
        except IsADirectoryError:
            raise WorkspaceError(f"{path!r} is a directory. Use list_files instead.") from None

        return ''.join(lines)


    async def write(self, path: str, content: str) -> None:
        '''Create or replace a file. Parent directories are created as needed.'''

        target = self.resolve(path)

        if target.is_dir():
            raise WorkspaceError(f"{path!r} is a directory, not a file.")

        await asyncio.to_thread(_write_atomic, target, content)


    async def edit(self, path: str, old: str, new: str) -> None:
        '''Replace exactly one occurrence of old text with new text.'''

        content = await self.read(path)
        count = content.count(old)

        if count == 0:
            raise WorkspaceError("The specified text was not found in the file.")

        if count > 1:
            raise WorkspaceError(
                f"The specified text occurs {count} times; "
                "provide more context to uniquely identify the edit."
            )

        target = self.resolve(path)
        await asyncio.to_thread(_write_atomic, target, content.replace(old, new, 1))


    async def list(self, path: str = '.') -> list[str]:
        '''List a directory. Directories come back with a trailing slash.'''

        target = self.resolve(path)

        if not target.is_dir():
            raise WorkspaceError(f"{path!r} is not a directory.")

        return await asyncio.to_thread(_list_dir, target)


    async def search(self, query: str, path: str = '.', max_results: int = 50) -> str:
        '''Search files recursively for a text or regex pattern.'''

        relative = self.resolve(path).relative_to(self.root)

        try:
            process = await asyncio.create_subprocess_exec(
                "rg",
                "--line-number",
                "--with-filename",
                "--color=never",
                "--no-messages",
                # Bounds any one pathological file. rg's own --max-count is per
                # file, so the total the model sees is capped below instead.
                "--max-count", str(max_results),
                "-e", query,
                "--", str(relative) if relative.parts else ".",
                cwd=self.root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "ripgrep (rg) is required but was not found. "
                "Please install ripgrep."
            ) from None

        stdout, stderr = await process.communicate()

        if process.returncode == 1:
            return "No matches found."

        if process.returncode != 0 and not stdout:
            raise RuntimeError(f"Search failed: {stderr.decode().strip()}")

        lines = stdout.decode(errors="replace").splitlines()

        if len(lines) > max_results:
            lines = lines[:max_results]
            lines.append(f"... truncated at {max_results} matches.")

        return '\n'.join(lines)


    def _decode(self, line: bytes, path: str) -> str:
        # Strict on purpose. Decoding leniently would let edit() write
        # replacement characters back over the bytes it never understood.
        try:
            text = line.decode()
        except UnicodeDecodeError:
            raise WorkspaceError(f"{path!r} is not valid UTF-8 text.") from None

        if '\x00' in text:
            raise WorkspaceError(f"{path!r} looks like a binary file.")

        return text


def _write_atomic(target: Path, content: str) -> None:
    '''Write content so readers see either the old file or the new one.'''

    target.parent.mkdir(parents=True, exist_ok=True)

    # Same directory as the target, so the rename stays on one filesystem.
    handle, temp_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".harness"
    )
    temp_path = Path(temp_name)

    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())

        try:
            shutil.copymode(target, temp_path)
        except FileNotFoundError:
            # New file: mkstemp made it 0600, a surprising mode for source code
            # to pick up just by being written through us.
            temp_path.chmod(NEW_FILE_MODE)

        os.replace(temp_path, target)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _list_dir(target: Path) -> list[str]:
    entries = []

    for entry in sorted(target.iterdir(), key=lambda path: path.name):
        if entry.name in DENIED_NAMES:
            continue
        entries.append(f"{entry.name}/" if entry.is_dir() else entry.name)

    return entries
