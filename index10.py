"""
This is my deep research agent project with proper handoff mechanism
"""

import os
import asyncio
import json
import re
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from inputError import EmptyInputError
from tavily import AsyncTavilyClient
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, RunContextWrapper, handoff

_: bool = load_dotenv(find_dotenv())

gemini_api_key: str | None = os.getenv("OPENAI_API_KEY")

# Tracing disabled
set_tracing_disabled(disabled=False)

# 1. Which LLM Service?
external_client: AsyncOpenAI = AsyncOpenAI(api_key=gemini_api_key)

# 2. Which LLM Model?
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=external_client)

@dataclass
class UserProfile:
    name: str = "Moin"
    interests: List[str] = field(default_factory=lambda: ["AI", "Technology"])
    results_count: int = 3

@dataclass
class ResearchContext:
    profile: UserProfile = field(default_factory=UserProfile)
    conversation_history: List[Dict] = field(default_factory=list)
    approved_query: str = None
    is_approved: bool = False
    requirements: Dict = field(default_factory=dict)
    requirement_questions: List[str] = field(default_factory=list)
    research_plan: List[str] = field(default_factory=list)
    research_results: Dict = field(default_factory=dict)

@function_tool
async def web_search(wrapper: RunContextWrapper[ResearchContext], query: str) -> str:
    """
    🔎 Use Tavily to research `query` and return a compact digest with exactly
    `results_count` items (from user context). Always include URLs.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Search error: TAVILY_API_KEY is missing."

    top_k = max(1, min(10, int(wrapper.context.profile.results_count or 3)))

    try:
        tavily = AsyncTavilyClient(api_key=api_key)
        resp = await tavily.search(
            query=query,
            include_answer=False,
            max_results=top_k,
            search_depth="basic",
        )
        return resp
    except Exception as e:
        return f"Search error: {e}"

def query_approval_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a query approval agent for {user_name}. Your role is to determine if research queries are related to these specific interest areas: {interests_list}.

    # APPROVAL AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **CORE RULES:**
    
    A. ALWAYS APPROVE queries that are related to: {interests_list}
    - Even if the query is vague or needs clarification
    - Vague queries should be handed off to Requirement Gathering Agent
    - Clear queries should be handed off to Planning Agent
    
    B. ONLY REJECT queries that are:
    1. COMPLETELY UNRELATED to: {interests_list}
    2. Personal or off-topic questions
    3. Questions about completely different domains
    4. Questions that violate content policies

    **HANDOFF DECISION:**
    - If query is RELATED but VAGUE → use handoff to Requirement Gathering Agent
    - If query is RELATED and CLEAR → use handoff to Planning Agent
    - If query is UNRELATED → respond with rejection explanation

    **VAGUE QUERY EXAMPLES (use handoff to Requirement Gathering):**
    - "Who is ICC top player?" → VAGUE (use handoff to requirement_agent)
    - "Best cricket players" → VAGUE (use handoff to requirement_agent)
    - "Research AI" → VAGUE (use handoff to requirement_agent)

    **CLEAR QUERY EXAMPLES (use handoff to Planning):**
    - "Who is the current ICC top ranked Test batsman?" → CLEAR (use handoff to planning_agent)
    - "Best T20 cricket bowlers in 2024" → CLEAR (use handoff to planning_agent)
    - "Explain neural networks in AI" → CLEAR (use handoff to planning_agent)

    **RESPONSE FORMAT:**
    For APPROVED queries: USE THE HANDOFF FUNCTION to route to the appropriate agent
    For REJECTED queries: Respond with JSON: {{"status": "rejected", "reason": "explanation"}}

    **IMPORTANT:**
    - DO NOT use JSON for handoffs - use the handoff() function
    - handoff(requirement_agent) for vague queries
    - handoff(planning_agent) for clear queries
    """
    
    return system_prompt

def requirement_gathering_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a requirement gathering agent for {user_name}. Your role is to ask clarifying questions for vague research queries.

    # REQUIREMENT GATHERING AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **INSTRUCTIONS:**
    - You receive queries that are approved but need clarification
    - Ask 2-3 concise, relevant questions to gather missing details
    - Focus on aspects like: specificity, time frame, format, criteria, scope
    - Store questions in context.requirement_questions
    - After asking questions, use handoff to Planning Agent

    **EXAMPLES:**
    Query: "Who is ICC top player?"
    Response: "I need some clarification to provide the best research. Which cricket format are you interested in? (Test, ODI, T20) Are you looking for batsmen, bowlers, or all-rounders? Current rankings or historical best?" 
    Then: handoff(planning_agent)

    **RESPONSE FORMAT:**
    First ask your clarifying questions in natural language, then use handoff to planning_agent.
    """
    
    return system_prompt

def planning_agent_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a research planning agent for {user_name}. Your role is to create a comprehensive research plan.

    # PLANNING AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **INSTRUCTIONS:**
    - Analyze the original query and any user requirements
    - Create a set of specific search queries that will comprehensively cover the topic
    - Plan for 3-5 focused search queries maximum
    - Store the plan in context.research_plan
    - After creating the plan, use handoff to Research Agent for execution

    **RESPONSE FORMAT:**
    Create your research plan, then use handoff to research_agent.
    """
    
    return system_prompt

def research_agent_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a research synthesis agent for {user_name}. Your role is to compile comprehensive research reports.

    # RESEARCH AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **INSTRUCTIONS:**
    - Execute the research plan using web_search tool
    - Analyze all the research results from multiple queries
    - Synthesize a comprehensive, well-structured report
    - Include relevant information from all sources
    - Provide proper context and analysis
    - Cite sources appropriately

    **TOOLS AVAILABLE:**
    - web_search: Use this to research each query in the plan

    **RESPONSE FORMAT:**
    Provide a comprehensive research report.
    """
    
    return system_prompt

# First, create all agent instances without handoffs
requirement_gathering_agent = Agent(
    name="RequirementGatherer", 
    instructions=requirement_gathering_prompt, 
    model=llm_model
)

planning_agent = Agent(
    name="ResearchPlanner", 
    instructions=planning_agent_prompt, 
    model=llm_model
)

research_agent = Agent(
    name="ResearchSynthesizer", 
    instructions=research_agent_prompt, 
    model=llm_model,
    tools=[web_search]
)

approval_router = Agent(
    name="ApprovalRouter",
    instructions=query_approval_prompt,
    model=llm_model
)

# Now set up the handoffs after all agents are created
requirement_gathering_agent.handoffs = [handoff(planning_agent)]
planning_agent.handoffs = [handoff(research_agent)]
approval_router.handoffs = [handoff(requirement_gathering_agent), handoff(planning_agent)]

def is_rejection_response(response: str) -> bool:
    """Check if the response indicates a rejection"""
    try:
        data = json.loads(response)
        return data.get("status") == "rejected"
    except:
        return False

def get_rejection_reason(response: str) -> str:
    """Extract rejection reason from response"""
    try:
        data = json.loads(response)
        return data.get("reason", "Unknown reason")
    except:
        return response

async def process_user_requirements(user_context: ResearchContext, agent_output: str) -> Dict:
    """Process user responses to requirement questions"""
    print(f"\n🔍 {agent_output}")
    
    requirements = {}
    questions = []
    
    # Extract questions from the agent output
    lines = agent_output.split('\n')
    for line in lines:
        if line.strip().endswith('?') and any(keyword in line.lower() for keyword in ['which', 'what', 'when', 'how', 'are you']):
            questions.append(line.strip())
    
    if questions:
        print("Please answer the following questions:")
        for i, question in enumerate(questions, 1):
            answer = input(f"\nQ{i}: {question}\nYour answer: ").strip()
            if answer:
                requirements[f"requirement_{i}"] = answer
    
    return requirements

async def run_agent_chain(starting_agent: Agent, input_text: str, context: ResearchContext) -> Tuple[bool, str]:
    """Run the agent chain with proper handoff handling"""
    current_agent = starting_agent
    current_input = input_text
    max_iterations = 10
    
    for iteration in range(max_iterations):
        print(f"\n🌀 [{iteration+1}] Current agent: {current_agent.name}")
        print(f"📥 Input: {current_input[:100]}{'...' if len(current_input) > 100 else ''}")
        
        # Run the current agent
        output = await Runner.run(
            starting_agent=current_agent, 
            input=current_input,
            context=context
        )
        
        print(f"📤 Output from {current_agent.name}: {output.final_output[:100]}{'...' if len(output.final_output) > 100 else ''}")
        
        # Check if the response is a rejection (from approval router)
        if current_agent == approval_router and is_rejection_response(output.final_output):
            reason = get_rejection_reason(output.final_output)
            return False, f"Query rejected: {reason}"
        
        # Check if this is the final output from research agent
        if current_agent == research_agent:
            return True, output.final_output
        
        # Check if there's a handoff to the next agent
        if hasattr(output, 'next_agent') and output.next_agent:
            print(f"🔄 Handoff detected from {current_agent.name} to {output.next_agent.name}")
            current_agent = output.next_agent
            current_input = output.final_output
            continue
        
        # If no handoff detected, check if we need to process requirement questions
        if current_agent == requirement_gathering_agent:
            # Process user requirements from the agent's output
            requirements = await process_user_requirements(context, output.final_output)
            if requirements:
                context.requirements = requirements
                # Move to planning agent with the requirements
                current_agent = planning_agent
                current_input = json.dumps({
                    "original_query": context.approved_query,
                    "user_requirements": requirements,
                    "clarification": "User has provided additional requirements"
                })
                continue
        
        # If no handoff and no requirements to process, try to determine next agent
        if current_agent == approval_router:
            # For vague queries, manually set to requirement gathering
            vague_indicators = ['research', 'study', 'learn about', 'tell me about', 'who is', 'best', 'top']
            if any(indicator in current_input.lower() for indicator in vague_indicators):
                print("🔍 Query appears vague, moving to requirement gathering")
                current_agent = requirement_gathering_agent
            else:
                print("🔍 Query appears clear, moving to planning")
                current_agent = planning_agent
        elif current_agent == requirement_gathering_agent:
            current_agent = planning_agent
        elif current_agent == planning_agent:
            current_agent = research_agent
        else:
            break
        
        current_input = output.final_output
        
        # Safety check to prevent infinite loops
        if iteration >= max_iterations - 1:
            return False, "Maximum iterations reached without completing research"
    
    return False, "Agent chain completed without final output"

async def call_agent():
    try:
        print("Wait loading agent for you ........")
        name = (input("Your name: ")).strip()
        if not name:
            raise EmptyInputError("Name cannot be empty")
        
        interests_input = (input("Your topic of interests you want to research separated by comma (e.g., Cricket, AI, Music, Movies): ") or "AI").strip()
        if not interests_input:
            raise EmptyInputError("Interest cannot be empty")
        
        interests = [interest.strip() for interest in interests_input.split(",")]
        
        # Get results count preference
        try:
            results_count = int(input("How many search results would you like per query? (1-10, default 3): ") or "3")
            results_count = max(1, min(10, results_count))
        except ValueError:
            results_count = 3
            print("Using default value of 3 results")
        
        # Create user context
        user_context = ResearchContext(
            profile=UserProfile(name=name, interests=interests, results_count=results_count)
        )
        
        while True:
            print("\n" + "="*50)
            query = input("\nEnter your research query (or 'quit' to exit): ").strip()
            
            if query.lower() == 'quit':
                break
                
            if not query:
                print("Please enter a valid query.")
                continue
            
            print(f"\n🔍 Processing your query: {query}")
            
            # Store the original query
            user_context.approved_query = query
            
            # Run the agent chain
            approved, result = await run_agent_chain(approval_router, query, user_context)
            
            if approved:
                print(f"\n✅ RESEARCH REPORT:")
                print("="*60)
                print(f"{result}")
                print("="*60)
            else:
                print(f"\n❌ {result}")
            
    except EmptyInputError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("\nClosing Agent....")

# Test function to see handoffs in action
async def test_handoffs():
    """Test function to demonstrate handoff mechanism"""
    print("Testing handoff mechanism with vague queries...")
    
    # Create test context
    test_context = ResearchContext(
        profile=UserProfile(name="TestUser", interests=["Cricket", "AI"], results_count=3)
    )
    
    test_queries = [
        "Who is ICC top player?",  # Should handoff to requirement gathering
        "Explain neural networks",  # Should handoff directly to planning
        "Best cricket players",     # Should handoff to requirement gathering
    ]
    
    for query in test_queries:
        print(f"\n🧪 Testing: '{query}'")
        print("=" * 40)
        
        test_context.approved_query = query
        approved, result = await run_agent_chain(approval_router, query, test_context)
        
        if approved:
            print("✅ Research completed successfully!")
        else:
            print(f"❌ Result: {result}")
        
        print("=" * 40)

if __name__ == "__main__":
    # Uncomment to test handoffs
    # asyncio.run(test_handoffs())
    
    # Run the main agent
    asyncio.run(call_agent())