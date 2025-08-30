from .base import base_agent
from prompts.citation import citation_agent_prompt

citation_agent= base_agent.clone(
    name="CitationAgent", 
    instructions=citation_agent_prompt, 
)