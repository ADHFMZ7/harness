# tools.py
# registry of tools agent can access

from typing import Any, Callable, Mapping
from dataclasses import dataclass

'''
Tools I eventually want to support
- File i/o
- Web search
'''
# Types of tools?

@dataclass
class Tool:
    name: str
    description: str
    function: Callable

    def __call__(self, **kwargs: Any) -> Any:
        return self.function(**kwargs)

@dataclass 
class ToolCall:
    name: str
    arguments: Mapping[str, Any]

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

    def get_tools(self, provider: str): #Figure out type hints
        if provider == 'ollama':
            return [tool.function for tool in self.tools.values()]
        else:
            print("Invalid provider")


registry = ToolRegistry()

@registry.register
def add(a: int, b: int) -> int:
    '''takes adds two integers a and b and returns their sum'''
    print("Invoking tool add")
    return a + b
