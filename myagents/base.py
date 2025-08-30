import os
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, Agent

api_key: str | None = os.getenv("OPENAI_API_KEY")

# Enable tracing to see handoffs
set_tracing_disabled(disabled=False)

# 1. Which LLM Service?
external_client: AsyncOpenAI = AsyncOpenAI(api_key=api_key)

# 2. Which LLM Model?
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=external_client)

# My base agent I'll extend this agent inorder to clone other agents.
base_agent = Agent(
    name="BaseAgent", 
    instructions="You are a helpful agent.", 
    model=llm_model
)