import pytest

from harness.tools import registry

LINES = [f"line{i}\n" for i in range(1, 11)]


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("".join(LINES))
    return str(path)


def test_registry_exposes_expected_tools():
    names = {tool.name for tool in registry.get_tools()}
    assert {"read_file", "write_file", "edit_file", "list_files"} <= names


def test_unknown_tool_raises_keyerror():
    with pytest.raises(KeyError):
        registry["nope"]


async def test_reads_whole_file(sample):
    assert await registry["read_file"](path=sample) == "".join(LINES)


async def test_range_is_inclusive_on_both_ends(sample):
    result = await registry["read_file"](path=sample, start_line=3, end_line=5)
    assert result == "line3\nline4\nline5\n"


async def test_start_line_without_end_line(sample):
    result = await registry["read_file"](path=sample, start_line=8)
    assert result == "line8\nline9\nline10\n"
