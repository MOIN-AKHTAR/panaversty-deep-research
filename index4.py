"""
This is my deep research agent project with proper handoff between agents
"""

import os
import asyncio
import json
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from inputError import EmptyInputError
from tavily import AsyncTavilyClient
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, RunContextWrapper, handoff

_: bool = load_dotenv(find_dotenv())

gemini_api_key: str | None = os.getenv("OPENAI_API_KEY")

# Tracing disabled
set_tracing_disabled(disabled=True)

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
    You are a query approval agent for {user_name}. Your ONLY role is to determine if research queries fall within these specific interest areas: {interests_list}.

    # APPROVAL AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **CORE RULES:**
    
    A. APPROVE these types of questions:
    1. Substantive research questions DIRECTLY related to: {interests_list}
    
    B. REJECT these types of questions:
    1. Substantive questions OUTSIDE of: {interests_list}
    2. General knowledge questions unrelated to the user's interests
    3. Meta-questions about yourself, the user, or the system

    **HANDOFF RULES:**
    - If query is APPROVED and needs clarification → handoff to Requirement Gathering Agent
    - If query is APPROVED and clear → handoff to Planning Agent
    - If query is REJECTED → respond with rejection reason

    **RESPONSE FORMAT:**
    For approved queries that need handoff, use the handoff function.
    For rejected queries, respond with a clear explanation.
    """
    
    return system_prompt

def requirement_gathering_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a requirement gathering agent for {user_name}. Your role is to ask clarifying questions to better understand research queries.

    # REQUIREMENT GATHERING AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **INSTRUCTIONS:**
    - Analyze the approved research query
    - Identify what specific information is needed to conduct effective research
    - Ask concise, relevant questions to gather missing details
    - Focus on aspects like: scope, depth, specific angles, time frame, etc.
    - Ask only the most essential questions (1-3 questions max)
    - After gathering requirements, handoff to Planning Agent

    **RESPONSE FORMAT:**
    Ask clarifying questions in a conversational manner.
    After receiving answers, use handoff to Planning Agent.
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
    - Analyze the original query and user requirements
    - Create a set of specific search queries that will comprehensively cover the topic
    - Consider different angles and aspects mentioned in the requirements
    - Plan for 3-5 focused search queries maximum
    - After creating the plan, handoff to Research Agent for execution

    **RESPONSE FORMAT:**
    Create a research plan and then handoff to Research Agent.
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
    Provide a comprehensive research report including:
    1. Executive summary of findings
    2. Detailed analysis of each major aspect
    3. Comparative insights where relevant
    4. Sources and references
    5. Conclusions and potential next steps
    """
    
    return system_prompt

# Create the specialized agents
requirement_gathering_agent: Agent = Agent(
    name="RequirementGatherer", 
    instructions=requirement_gathering_prompt, 
    model=llm_model,
    handoffs=[]  # This agent will handoff to planning_agent after gathering requirements
)

planning_agent: Agent = Agent(
    name="ResearchPlanner", 
    instructions=planning_agent_prompt, 
    model=llm_model,
    handoffs=[]  # This agent will handoff to research_agent after creating plan
)

research_agent: Agent = Agent(
    name="ResearchSynthesizer", 
    instructions=research_agent_prompt, 
    model=llm_model,
    tools=[web_search]
)

# Configure handoffs for requirement and planning agents
requirement_gathering_agent.handoffs = [handoff(planning_agent)]
planning_agent.handoffs = [handoff(research_agent)]

# Create the approval router agent with handoffs
approval_router: Agent = Agent(
    name="ApprovalRouter",
    instructions=query_approval_prompt,
    model=llm_model,
    handoffs=[handoff(requirement_gathering_agent), handoff(planning_agent)]
)

async def process_user_requirements(user_context: ResearchContext, questions: List[str]) -> Dict:
    """Process user responses to requirement questions"""
    requirements = {}
    
    print(f"\n🔍 To provide the best research, I need some details:")
    
    for i, question in enumerate(questions, 1):
        answer = input(f"\nQ{i}: {question}\nYour answer: ").strip()
        if answer:
            requirements[f"requirement_{i}"] = answer
    
    return requirements

async def call_agent():
    try:
        print("Wait loading agent for you ........")
        name = (input("Your name: ")).strip()
        if not name:
            raise EmptyInputError("Name cannot be empty")
        
        interests_input = (input("Your topic of interests you want to research separated by comma (AI, Politics etc.): ") or "AI").strip()
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
            
            # Start with the approval router
            current_agent = approval_router
            current_input = query
            
            while current_agent:
                print(f"\n🤖 {current_agent.name} is processing your request...")
                
                # Run the current agent
                output = await Runner.run(
                    starting_agent=current_agent, 
                    input=current_input,
                    context=user_context
                )

                print("OUTPUT -----> ")
                print(output)
                
                # Check if the agent handed off to another agent
                if output.next_agent:
                    print(f"↳ Handing off to {output.next_agent.name}...")
                    current_agent = output.next_agent
                    current_input = output.final_output
                else:
                    # Check if we need to process requirement questions
                    if (current_agent == requirement_gathering_agent and 
                        hasattr(user_context, 'requirement_questions') and 
                        user_context.requirement_questions):
                        
                        # Process user requirements
                        requirements = await process_user_requirements(
                            user_context, user_context.requirement_questions
                        )
                        user_context.requirements = requirements
                        
                        # Continue to planning agent with the requirements
                        current_agent = planning_agent
                        current_input = json.dumps({
                            "original_query": user_context.approved_query,
                            "user_requirements": requirements
                        })
                    else:
                        # Final output from research agent
                        print(f"\n✅ RESEARCH REPORT:")
                        print("="*60)
                        print(f"{output.final_output}")
                        print("="*60)
                        break
            
    except EmptyInputError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("\nClosing Agent....")

# For testing the handoff mechanism without user input
async def test_handoff():
    """Test function to demonstrate the handoff mechanism"""
    print("Testing handoff mechanism...")
    
    # Create test context
    test_context = ResearchContext(
        profile=UserProfile(name="TestUser", interests=["AI", "Technology"], results_count=3)
    )
    
    # Test query
    test_query = "Research recent advancements in AI"
    
    print(f"\nTesting with query: {test_query}")
    
    # Start with approval router
    current_agent = approval_router
    current_input = test_query
    
    max_iterations = 10  # Safety limit
    iteration = 0
    
    while current_agent and iteration < max_iterations:
        iteration += 1
        print(f"\n[{iteration}] {current_agent.name} processing...")
        
        output = await Runner.run(
            starting_agent=current_agent, 
            input=current_input,
            context=test_context
        )
        
        if output.next_agent:
            print(f"  ↳ Handing off to: {output.next_agent.name}")
            current_agent = output.next_agent
            current_input = output.final_output
        else:
            print(f"  ✅ Final output from {current_agent.name}")
            print(f"  Output: {output.final_output[:200]}...")
            break
    
    if iteration >= max_iterations:
        print("  ⚠️  Maximum iterations reached")

if __name__ == "__main__":
    # Uncomment the next line to test the handoff mechanism
    # asyncio.run(test_handoff())
    
    # Run the main agent
    asyncio.run(call_agent())