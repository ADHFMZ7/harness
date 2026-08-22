# main.py

from agent import Agent
from llm import OllamaLLM
from tools import registry

def main():

    llm = OllamaLLM('qwen3.5:9b')

    agent = Agent(llm, registry)

    while (prompt := input(">> ")) != "exit":
        for response in agent.run(prompt):
            print(response)


if __name__ == '__main__':
    main()

