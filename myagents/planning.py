from .base import base_agent
from prompts.planning import planning_agent_prompt
from .research import research_agent
from agents import handoff
from agents.extensions import handoff_filters

planning_agent = base_agent.clone(
    name="ResearchPlanner", 
    instructions=planning_agent_prompt, 
    # model=llm_model,
    handoffs=[handoff(research_agent, tool_name_override="ResearchAgent", tool_description_override="A research synthesis agent that executes search queries, gathers and cites current information, and produces comprehensive, well-structured reports using specialized tools for accuracy and quality.", input_filter=handoff_filters.remove_all_tools)] 
)