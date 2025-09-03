from .base import base_agent
from prompts.research import research_agent_prompt
from tools.websearch import web_search
from .citation import citation_agent
from .reflection import reflection_agent
from agents import ModelSettings


research_agent = base_agent.clone(
    name="ResearchAgent", 
    instructions=research_agent_prompt, 
    model_settings = ModelSettings(
        max_tokens=1000,
        parallel_tool_calls=True,
        temperature=0.3,
    ),
    tools=[ 
        web_search, 
        citation_agent.as_tool(tool_name="CitationAgent",tool_description="Generates proper academic citations and ensures all research sources are properly attributed with consistent formatting."),
        reflection_agent.as_tool(tool_name="RflectionAgent", tool_description="Provides critical analysis of research quality, identifies gaps and biases, and suggests improvements for completeness and accuracy.")
        ], 
)