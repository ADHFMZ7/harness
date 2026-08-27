import shutil

import pytest

from harness import workspace as workspace_module
from harness.workspace import (
    HostWorkspace,
    PathDenied,
    PathOutsideWorkspace,
    WorkspaceError,
)

LINES = [f"line{i}\n" for i in range(1, 11)]

needs_ripgrep = pytest.mark.skipif(
    shutil.which("rg") is None, reason="ripgrep is not installed"
)


@pytest.fixture
def outside(tmp_path):
    """A file the agent must never be able to reach."""
    path = tmp_path / "outside.txt"
    path.write_text("secret\n")
    return path


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    return HostWorkspace(root)


@pytest.fixture
def sample(ws):
    path = ws.root / "sample.txt"
    path.write_text("".join(LINES))
    return "sample.txt"


# ---------------------------------------------------------------- confinement

def test_root_is_resolved(tmp_path):
    """A root reached through a symlink still matches paths resolved under it."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)

    ws = HostWorkspace(link)

    assert ws.root == real.resolve()
    assert ws.resolve("a.txt") == real.resolve() / "a.txt"


def test_absolute_path_is_rejected(ws, outside):
    with pytest.raises(PathOutsideWorkspace):
        ws.resolve(str(outside))


def test_parent_traversal_is_rejected(ws):
    with pytest.raises(PathOutsideWorkspace):
        ws.resolve("../outside.txt")


def test_traversal_hidden_mid_path_is_rejected(ws):
    with pytest.raises(PathOutsideWorkspace):
        ws.resolve("sub/../../outside.txt")


def test_home_relative_path_is_rejected(ws):
    with pytest.raises(PathOutsideWorkspace):
        ws.resolve("~/.ssh/id_rsa")


def test_symlinked_file_escaping_the_root_is_rejected(ws, outside):
    (ws.root / "link.txt").symlink_to(outside)

    with pytest.raises(PathOutsideWorkspace):
        ws.resolve("link.txt")


def test_symlinked_directory_escaping_the_root_is_rejected(ws, outside):
    (ws.root / "escape").symlink_to(outside.parent)

    with pytest.raises(PathOutsideWorkspace):
        ws.resolve("escape/outside.txt")


def test_symlink_staying_inside_the_root_is_allowed(ws):
    (ws.root / "real.txt").write_text("hi\n")
    (ws.root / "alias.txt").symlink_to(ws.root / "real.txt")

    assert ws.resolve("alias.txt") == ws.root / "real.txt"


def test_git_directory_is_denied(ws):
    with pytest.raises(PathDenied):
        ws.resolve(".git/config")


def test_nested_git_directory_is_denied(ws):
    with pytest.raises(PathDenied):
        ws.resolve("vendor/dep/.git/hooks/pre-commit")


def test_empty_and_null_paths_are_rejected(ws):
    with pytest.raises(WorkspaceError):
        ws.resolve("   ")
    with pytest.raises(WorkspaceError):
        ws.resolve("a\x00b")


def test_root_itself_resolves(ws):
    assert ws.resolve(".") == ws.root


async def test_reads_do_not_escape_the_root(ws, outside):
    (ws.root / "link.txt").symlink_to(outside)

    with pytest.raises(PathOutsideWorkspace):
        await ws.read("link.txt")


async def test_writes_do_not_escape_the_root(ws, outside):
    with pytest.raises(PathOutsideWorkspace):
        await ws.write("../outside.txt", "clobbered")

    assert outside.read_text() == "secret\n"


# ----------------------------------------------------------------------- read

async def test_reads_whole_file(ws, sample):
    assert await ws.read(sample) == "".join(LINES)


async def test_range_is_inclusive_on_both_ends(ws, sample):
    assert await ws.read(sample, start_line=3, end_line=5) == "line3\nline4\nline5\n"


async def test_start_line_without_end_line(ws, sample):
    assert await ws.read(sample, start_line=8) == "line8\nline9\nline10\n"


async def test_missing_file_reports_the_path_it_was_given(ws):
    with pytest.raises(WorkspaceError, match="nope.txt"):
        await ws.read("nope.txt")


async def test_directory_is_not_readable(ws):
    (ws.root / "sub").mkdir()

    with pytest.raises(WorkspaceError):
        await ws.read("sub")


async def test_binary_file_is_refused(ws):
    (ws.root / "blob.bin").write_bytes(b"\x89PNG\x00\x1a\n" * 100)

    with pytest.raises(WorkspaceError):
        await ws.read("blob.bin")


async def test_binary_file_without_newlines_is_refused(ws):
    (ws.root / "blob.bin").write_bytes(b"\x00" * 200_000)

    with pytest.raises(WorkspaceError):
        await ws.read("blob.bin")


async def test_non_utf8_text_is_refused_rather_than_mangled(ws):
    (ws.root / "latin.txt").write_bytes(b"caf\xe9 au lait\n")

    with pytest.raises(WorkspaceError):
        await ws.read("latin.txt")


async def test_read_stops_at_the_byte_cap(ws, sample, monkeypatch):
    monkeypatch.setattr(workspace_module, "MAX_READ_BYTES", 12)

    with pytest.raises(WorkspaceError):
        await ws.read(sample)


async def test_a_narrow_range_still_reads_from_a_capped_file(ws, sample, monkeypatch):
    monkeypatch.setattr(workspace_module, "MAX_READ_BYTES", 12)

    assert await ws.read(sample, start_line=1, end_line=1) == "line1\n"


# ---------------------------------------------------------------------- write

async def test_write_creates_missing_parents(ws):
    await ws.write("a/b/c.txt", "deep\n")

    assert (ws.root / "a/b/c.txt").read_text() == "deep\n"


async def test_write_replaces_content(ws, sample):
    await ws.write(sample, "replaced\n")

    assert (ws.root / sample).read_text() == "replaced\n"


async def test_write_keeps_the_mode_of_an_existing_file(ws, sample):
    (ws.root / sample).chmod(0o600)

    await ws.write(sample, "replaced\n")

    assert (ws.root / sample).stat().st_mode & 0o777 == 0o600


async def test_new_files_are_not_created_private(ws):
    await ws.write("fresh.txt", "hi\n")

    assert (ws.root / "fresh.txt").stat().st_mode & 0o777 == 0o644


async def test_write_leaves_no_temporary_files_behind(ws, sample):
    await ws.write(sample, "replaced\n")

    assert sorted(p.name for p in ws.root.iterdir()) == ["sample.txt"]


async def test_write_replaces_the_target_of_an_internal_symlink(ws):
    (ws.root / "real.txt").write_text("old\n")
    (ws.root / "alias.txt").symlink_to(ws.root / "real.txt")

    await ws.write("alias.txt", "new\n")

    assert (ws.root / "real.txt").read_text() == "new\n"
    assert (ws.root / "alias.txt").is_symlink()


async def test_cannot_write_over_a_directory(ws):
    (ws.root / "sub").mkdir()

    with pytest.raises(WorkspaceError):
        await ws.write("sub", "nope")


# ----------------------------------------------------------------------- edit

async def test_edit_replaces_a_unique_occurrence(ws, sample):
    await ws.edit(sample, "line4", "LINE4")

    assert "LINE4\n" in (ws.root / sample).read_text()


async def test_edit_refuses_when_absent(ws, sample):
    with pytest.raises(WorkspaceError, match="not found"):
        await ws.edit(sample, "nowhere", "x")


async def test_edit_refuses_when_ambiguous(ws):
    (ws.root / "dup.txt").write_text("a\na\n")

    with pytest.raises(WorkspaceError, match="occurs 2 times"):
        await ws.edit("dup.txt", "a", "b")


async def test_a_refused_edit_leaves_the_file_alone(ws, sample):
    with pytest.raises(WorkspaceError):
        await ws.edit(sample, "nowhere", "x")

    assert (ws.root / sample).read_text() == "".join(LINES)


# ----------------------------------------------------------------------- list

async def test_list_is_sorted_and_marks_directories(ws):
    (ws.root / "b.txt").touch()
    (ws.root / "a.txt").touch()
    (ws.root / "sub").mkdir()

    assert await ws.list() == ["a.txt", "b.txt", "sub/"]


async def test_list_hides_the_git_directory(ws):
    (ws.root / ".git").mkdir()
    (ws.root / "a.txt").touch()

    assert await ws.list() == ["a.txt"]


async def test_list_refuses_a_file(ws, sample):
    with pytest.raises(WorkspaceError):
        await ws.list(sample)


# --------------------------------------------------------------------- search

@needs_ripgrep
async def test_search_finds_matches_with_relative_paths(ws, sample):
    result = await ws.search("line4")

    assert "sample.txt:4:line4" in result


@needs_ripgrep
async def test_search_reports_no_matches(ws, sample):
    assert await ws.search("absent") == "No matches found."


@needs_ripgrep
async def test_search_treats_a_leading_dash_as_a_pattern(ws):
    (ws.root / "flags.txt").write_text("--count me\n")

    assert "flags.txt" in await ws.search("--count")


@needs_ripgrep
async def test_search_caps_total_results_across_files(ws):
    for i in range(10):
        (ws.root / f"f{i}.txt").write_text("needle\n")

    result = await ws.search("needle", max_results=4)
    lines = result.splitlines()

    assert len(lines) == 5
    assert lines[-1] == "... truncated at 4 matches."


@needs_ripgrep
async def test_search_cannot_escape_the_root(ws):
    with pytest.raises(PathOutsideWorkspace):
        await ws.search("secret", path="..")
