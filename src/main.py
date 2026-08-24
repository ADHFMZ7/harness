# main.py

import asyncio

from agent import Agent
from llm import OllamaLLM
from tools import registry


async def main():

    llm = OllamaLLM('qwen3.5:9b')

    agent = Agent(llm, registry)

    while (prompt := input(">> ")) != "exit":
        async for response in agent.run(prompt):
            print(response)


if __name__ == '__main__':
    asyncio.run(main())

