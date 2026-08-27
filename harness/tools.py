# tools.py
# registry of tools agent can access


from collections.abc import Callable
from typing import Any

from harness.models import Tool
from harness.workspace import Workspace

'''
Tools I eventually want to support
- Web search
'''

class ToolRegistry:

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def  register(
         self, 
         function: Callable[..., Any],
         *,
         name: str | None = None,
         description: str | None = None
    ) -> Callable[..., Any]:

        tool = Tool(name = name or function.__name__,
                    description=description or function.__doc__ or "",
                    function=function,
                    )

        self.tools[tool.name] = tool
        return function

    def get_tools(self) -> list[Tool]: 
        return list(self.tools.values())

    def __getitem__(self, key:str) -> Tool:
        tool = self.tools.get(key)
        if not tool:
            raise KeyError(f"Tool {key} does not exist")
        return tool


def build_registry(workspace: Workspace) -> ToolRegistry:
    '''Build the tool set the agent sees, bound to one workspace.

    The tools are deliberately thin. Confinement, atomic writes and the trash
    all live in the workspace, so there is one place to look for the rules and
    no way for a tool to acquire its own.
    '''

    registry = ToolRegistry()

    @registry.register
    def add(a: int, b: int) -> int:
        '''takes adds two integers a and b and returns their sum'''
        return a + b

    @registry.register
    async def list_files(dir_path: str = '.') -> list[str]:
        '''lists files in directory specified by path'''
        return await workspace.list(dir_path)

    @registry.register
    async def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        '''Read contents of a file. Returns chunk bounded by provided range. If no range is provided, entire file is read'''
        return await workspace.read(path, start_line, end_line)

    @registry.register
    async def search_file(query: str, path: str = ".", max_results: int = 50) -> str:
        """Search files recursively for a text or regex pattern."""
        return await workspace.search(query, path, max_results)

    @registry.register
    async def write_file(path: str, content: str) -> str:
        """Create or overwrite a file with the provided contents."""
        await workspace.write(path, content)
        return f"Successfully wrote {path}"

    @registry.register
    async def edit_file(path: str, old: str, new: str) -> str:
        """Replace exactly one occurrence of old text with new text."""
        await workspace.edit(path, old, new)
        return f"Successfully edited {path}"

    return registry
