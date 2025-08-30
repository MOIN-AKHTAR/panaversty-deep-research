from .base import base_agent
from prompts.reflection import reflection_agent_prompt

reflection_agent= base_agent.clone(
    name="ReflectionAgent", 
    instructions=reflection_agent_prompt,    
)