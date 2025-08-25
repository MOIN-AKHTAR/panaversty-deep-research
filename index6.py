"""
This is my deep research agent project with improved approval logic
"""

import os
import asyncio
import json
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Tuple, Optional, Union
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
    You are a query approval agent for {user_name}. Your role is to determine if research queries are related to these specific interest areas: {interests_list}.

    # APPROVAL AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **CORE RULES:**
    
    A. APPROVE these types of questions:
    1. Questions DIRECTLY related to: {interests_list}
    2. General knowledge questions ABOUT topics in: {interests_list}
    3. Technical questions related to the user's interests
    4. Questions about tools, technologies, or concepts in these fields
    
    B. REJECT these types of questions:
    1. Questions COMPLETELY UNRELATED to: {interests_list}
    2. Personal or off-topic questions
    3. Questions about completely different domains (sports, cooking, etc.)
    4. Questions that violate content policies

    **SPECIFIC GUIDELINES FOR {interests_list}:**
    - For DevOps interest: APPROVE questions about tools, technologies, infrastructure
    - For Coding interest: APPROVE questions about programming, languages, frameworks
    - For AI interest: APPROVE questions about algorithms, models, applications
    
    **HANDOFF DECISION:**
    - If query is APPROVED and needs clarification → handoff to Requirement Gathering Agent
    - If query is APPROVED and clear → handoff to Planning Agent
    - If query is REJECTED → respond with clear rejection explanation

    **RESPONSE FORMAT:**
    For approved queries: Use handoff function to route to appropriate agent
    For rejected queries: Respond with JSON: {{"status": "rejected", "reason": "explanation"}}

    **EXAMPLES:**
    Query: "What is nginx?" → APPROVED (related to DevOps/Infrastructure)
    Query: "Explain Python decorators" → APPROVED (related to Coding)
    Query: "What is reinforcement learning?" → APPROVED (related to AI)
    Query: "How to make pasta?" → REJECTED (completely unrelated)
    Query: "Who won the basketball game?" → REJECTED (completely unrelated)
    Query: "What is quantum physics?" → REJECTED (unless physics is in interests)
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
    - You will only receive queries that are APPROVED and need clarification
    - Analyze the approved research query
    - Identify what specific information is needed to conduct effective research
    - Ask concise, relevant questions to gather missing details
    - Focus on aspects like: scope, depth, specific angles, time frame, etc.
    - Ask only the most essential questions (1-3 questions max)
    - Store questions in context.requirement_questions
    - After gathering requirements, handoff to Planning Agent

    **RESPONSE FORMAT:**
    Respond with JSON: {{"questions": ["question1", "question2", ...]}}
    Then use handoff to Planning Agent.
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
    - Store the plan in context.research_plan
    - After creating the plan, handoff to Research Agent for execution

    **RESPONSE FORMAT:**
    Respond with JSON: {{"search_queries": ["query1", "query2", ...], "research_strategy": "brief explanation"}}
    Then use handoff to Research Agent.
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
    model=llm_model
)

planning_agent: Agent = Agent(
    name="ResearchPlanner", 
    instructions=planning_agent_prompt, 
    model=llm_model
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
        
        interests_input = (input("Your topic of interests you want to research separated by comma (AI, DevOps, Coding etc.): ") or "AI").strip()
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
                
                # Check if the response is a rejection
                if current_agent == approval_router and is_rejection_response(output.final_output):
                    reason = get_rejection_reason(output.final_output)
                    print(f"\n❌ Query rejected: {reason}")
                    break
                
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

# Test function to demonstrate improved approval logic
async def test_improved_approval():
    """Test function to demonstrate improved approval logic"""
    print("Testing improved approval logic for relevant queries...")
    
    # Create test context with DevOps/Coding/AI interests
    test_context = ResearchContext(
        profile=UserProfile(name="TestUser", interests=["AI", "DevOps", "Coding"], results_count=3)
    )
    
    # Test queries that should be APPROVED (related to interests)
    test_queries = [
        "What is nginx?",  # DevOps tool → SHOULD APPROVE
        "Explain Python decorators",  # Coding concept → SHOULD APPROVE
        "What is reinforcement learning?",  # AI concept → SHOULD APPROVE
        "How does Docker work?",  # DevOps tool → SHOULD APPROVE
        "What are neural networks?",  # AI concept → SHOULD APPROVE
        "Explain git branching strategy",  # Coding/DevOps → SHOULD APPROVE
    ]
    
    for query in test_queries:
        print(f"\n--- Testing query: '{query}' ---")
        
        output = await Runner.run(
            starting_agent=approval_router, 
            input=query,
            context=test_context
        )
        
        if is_rejection_response(output.final_output):
            reason = get_rejection_reason(output.final_output)
            print(f"❌ Incorrectly rejected: {reason}")
        else:
            print(f"✅ Correctly approved - proceeding to next agent")
            
            # Check if it's a handoff
            if output.next_agent:
                print(f"   Handing off to: {output.next_agent.name}")

if __name__ == "__main__":
    # Uncomment to test improved approval logic
    # asyncio.run(test_improved_approval())
    
    # Run the main agent
    asyncio.run(call_agent())