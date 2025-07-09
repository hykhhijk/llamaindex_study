from llama_index.core.memory import ChatMemoryBuffer
import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
#### Basic Setting

import  load_dotenv
load_dotenv.load_dotenv("../../All_LLM_tutorial/.env")

# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    tools=[multiply],
    llm=OpenAI(model="gpt-4o-mini"),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)
memory = ChatMemoryBuffer.from_defaults(token_limit=40000)

async def main():
    # Run the agent
    response =  await agent.run("What's previous answer of you", memory=memory)
    print(str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())
