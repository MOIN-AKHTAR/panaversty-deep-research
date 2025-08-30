from .base import base_agent
from prompts.approval import query_approval_prompt
from output.approval import ApproveAgentResponse
from agents import handoff
from .planning import planning_agent
from agents.extensions import handoff_filters

approval_router = base_agent.clone(
    name="ApprovalRouter",
    instructions=query_approval_prompt,
    # model=llm_model,
    output_type=ApproveAgentResponse,
    handoffs=[handoff(planning_agent, tool_name_override="PlanningAgent",tool_description_override="A research planning agent that analyzes user queries and generates 3-5 specific, current, actionable web search queries for comprehensive topic coverage before handing off to a research agent.",input_filter=handoff_filters.remove_all_tools)]    
)