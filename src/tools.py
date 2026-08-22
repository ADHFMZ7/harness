# tools.py
# registry of tools agent can access

from typing import Any, Callable
from models import Tool

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
    print("Invoking tool add")
    return a + b
