# tools.py
# registry of tools agent can access

from typing import Any, Callable
from models import Tool

import aiofiles
from aiofiles import os

'''
Tools I eventually want to support
- File i/o
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

registry = ToolRegistry()

@registry.register
def add(a: int, b: int) -> int:
    '''takes adds two integers a and b and returns their sum'''
    print(f"Tools: Adding numbers {a} {b}")
    return a + b


@registry.register
async def list_files(dir_path: str = '.') -> list[str]:
    '''lists files in directory specified by path'''
    print(f"Tools: Listing files at path {dir_path}")
    return await os.listdir(dir_path)


@registry.register
async def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> Any:
    '''Read contents of a file. Returns chunk bounded by provided range. If no range is provided, entire file is read'''

    print(f"Tools: Reading file at path {path}")

    lines = []

    async with aiofiles.open(path, mode='r') as file:
        
        line_num = 1
        while (line := await file.readline()):
            if end_line and line_num > end_line:
                break
            if start_line and line_num < start_line:
                continue 
            lines.append(line)

            line_num += 1

    return ''.join(lines)

# @registry.register
# async def search_file(query: str, path: str, max_results: int = 50) -> Any:













