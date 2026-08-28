import pytest

from harness.tools import build_registry
from harness.workspace import HostWorkspace, PathOutsideWorkspace

LINES = [f"line{i}\n" for i in range(1, 11)]


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "sample.txt").write_text("".join(LINES))
    return HostWorkspace(root)


@pytest.fixture
def registry(workspace):
    return build_registry(workspace)


def test_registry_exposes_expected_tools(registry):
    names = {tool.name for tool in registry.get_tools()}
    assert {"read_file", "write_file", "edit_file", "list_files"} <= names


def test_unknown_tool_raises_keyerror(registry):
    with pytest.raises(KeyError):
        registry["nope"]


def test_descriptions_come_from_docstrings(registry):
    assert registry["write_file"].description.startswith("Create or overwrite")


async def test_read_file_reads_through_the_workspace(registry):
    assert await registry["read_file"](path="sample.txt") == "".join(LINES)


async def test_read_file_honours_a_line_range(registry):
    result = await registry["read_file"](path="sample.txt", start_line=3, end_line=5)
    assert result == "line3\nline4\nline5\n"


async def test_write_file_lands_in_the_workspace(registry, workspace):
    await registry["write_file"](path="new.txt", content="written\n")

    assert (workspace.root / "new.txt").read_text() == "written\n"


async def test_edit_file_lands_in_the_workspace(registry, workspace):
    await registry["edit_file"](path="sample.txt", old="line4", new="LINE4")

    assert "LINE4\n" in (workspace.root / "sample.txt").read_text()


async def test_list_files_defaults_to_the_root(registry):
    assert await registry["list_files"]() == ["sample.txt"]


async def test_tools_cannot_be_talked_out_of_the_workspace(registry):
    with pytest.raises(PathOutsideWorkspace):
        await registry["read_file"](path="/etc/passwd")

    with pytest.raises(PathOutsideWorkspace):
        await registry["write_file"](path="../escape.txt", content="x")
