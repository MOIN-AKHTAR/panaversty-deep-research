"""
This is my deep research agent project with handoff between specialized agents
"""

import os
import asyncio
import json
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from inputError import EmptyInputError
from tavily import AsyncTavilyClient
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, RunContextWrapper

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

@function_tool
async def handoff_to_requirement_gathering(wrapper: RunContextWrapper[ResearchContext], query: str) -> str:
    """
    🤝 Handoff to requirement gathering agent to ask clarifying questions
    for the research query.
    """
    try:
        # Store the approved query
        wrapper.context.approved_query = query
        wrapper.context.is_approved = True
        
        # Call requirement gathering agent
        requirement_output = await Runner.run(
            starting_agent=requirement_gathering_agent, 
            input=query,
            context=wrapper.context
        )
        
        # Parse and store requirement questions
        requirement_data = json.loads(requirement_output.final_output)
        wrapper.context.requirement_questions = requirement_data.get("questions", [])
        
        return json.dumps({
            "status": "handoff_complete",
            "questions": wrapper.context.requirement_questions,
            "next_step": "await_user_responses"
        })
    except Exception as e:
        return f"Handoff error: {e}"

@function_tool
async def handoff_to_planning(wrapper: RunContextWrapper[ResearchContext], user_responses: Dict) -> str:
    """
    🤝 Handoff to planning agent to create a research plan based on
    user requirements and responses.
    """
    try:
        # Store user responses as requirements
        wrapper.context.requirements = user_responses
        
        # Call planning agent
        planning_output = await Runner.run(
            starting_agent=planning_agent, 
            input=json.dumps({
                "original_query": wrapper.context.approved_query,
                "user_requirements": user_responses
            }),
            context=wrapper.context
        )
        
        # Parse and store research plan
        planning_data = json.loads(planning_output.final_output)
        wrapper.context.research_plan = planning_data.get("search_queries", [])
        
        return json.dumps({
            "status": "planning_complete",
            "research_plan": wrapper.context.research_plan,
            "next_step": "execute_research"
        })
    except Exception as e:
        return f"Planning handoff error: {e}"

@function_tool
async def execute_research_plan(wrapper: RunContextWrapper[ResearchContext]) -> str:
    """
    🔬 Execute the research plan by running all search queries
    and compiling results.
    """
    try:
        research_results = {}
        
        for i, query in enumerate(wrapper.context.research_plan, 1):
            print(f"\n🔍 Researching: {query}")
            
            # Execute search for each query in the plan
            search_result = await web_search(wrapper, query)
            research_results[f"query_{i}"] = {
                "search_query": query,
                "results": search_result
            }
        
        # Store results in context
        wrapper.context.research_results = research_results
        
        # Call research agent to compile final report
        final_output = await Runner.run(
            starting_agent=research_agent, 
            input=json.dumps({
                "original_query": wrapper.context.approved_query,
                "user_requirements": wrapper.context.requirements,
                "research_results": research_results
            }),
            context=wrapper.context
        )
        
        return final_output.final_output
        
    except Exception as e:
        return f"Research execution error: {e}"

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

    **TOOLS AVAILABLE:**
    - handoff_to_requirement_gathering: Use this when a query is approved but needs clarification

    **RESPONSE FORMAT:**
    You must respond with ONLY a JSON object in this exact format:
    {{
        "approved": true/false,
        "reason": "brief explanation for approval/rejection",
        "query": "the original query text",
        "action": "handoff" or "reject"  // Use "handoff" for approved queries that need clarification
    }}

    **EXAMPLES:**
    Query: "Explain attention mechanisms in AI?"
    → {{"approved": true, "reason": "AI is within user interests", "query": "Explain attention mechanisms in AI?", "action": "handoff"}}

    Query: "Tell me about quantum physics"
    → {{"approved": false, "reason": "Quantum physics not in user interests", "query": "Tell me about quantum physics", "action": "reject"}}

    Query: "Research AI"
    → {{"approved": true, "reason": "AI is within user interests", "query": "Research AI", "action": "handoff"}}
    """
    
    return system_prompt

def requirement_gathering_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a requirement gathering agent for {user_name}. Your role is to ask clarifying questions to better understand research queries and gather necessary details.

    # REQUIREMENT GATHERING AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **INSTRUCTIONS:**
    - Analyze the approved research query
    - Identify what specific information is needed to conduct effective research
    - Ask concise, relevant questions to gather missing details
    - Focus on aspects like: scope, depth, specific angles, time frame, etc.
    - Ask only the most essential questions (1-3 questions max)

    **RESPONSE FORMAT:**
    You must respond with ONLY a JSON object in this exact format:
    {{
        "questions": ["question 1", "question 2", ...],
        "refined_query": "a more specific version of the original query based on what you understand"
    }}

    **EXAMPLES:**
    Original Query: "Research AI"
    → {{
        "questions": [
            "What specific aspect of AI are you most interested in? (e.g., machine learning, neural networks, AI ethics)",
            "What depth of information are you looking for? (introductory overview or technical details)"
        ],
        "refined_query": "Research current trends in artificial intelligence with focus on machine learning applications"
    }}
    """
    
    return system_prompt

def planning_agent_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a research planning agent for {user_name}. Your role is to create a comprehensive research plan based on the user's query and requirements.

    # PLANNING AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **INSTRUCTIONS:**
    - Analyze the original query and user requirements
    - Create a set of specific search queries that will comprehensively cover the topic
    - Consider different angles and aspects mentioned in the requirements
    - Plan for 3-5 focused search queries maximum
    - Ensure queries are specific enough to yield good results

    **TOOLS AVAILABLE:**
    - web_search: Available to the next agent who will execute these queries

    **RESPONSE FORMAT:**
    You must respond with ONLY a JSON object in this exact format:
    {{
        "search_queries": ["query 1", "query 2", "query 3", ...],
        "research_strategy": "brief explanation of your approach"
    }}

    **EXAMPLES:**
    Input: {{"original_query": "Research AI", "user_requirements": {{"aspect": "machine learning", "depth": "technical"}}}}
    → {{
        "search_queries": [
            "latest advancements in machine learning algorithms 2024",
            "neural network architectures recent research papers",
            "comparison of deep learning frameworks performance"
        ],
        "research_strategy": "Focusing on technical aspects of machine learning with recent advancements"
    }}
    """
    
    return system_prompt

def research_agent_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a research synthesis agent for {user_name}. Your role is to compile comprehensive research reports from multiple search results.

    # RESEARCH AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **INSTRUCTIONS:**
    - Analyze all the research results from multiple queries
    - Synthesize a comprehensive, well-structured report
    - Include relevant information from all sources
    - Provide proper context and analysis
    - Cite sources appropriately
    - Address all aspects of the original query and user requirements

    **TOOLS AVAILABLE:**
    - execute_research_plan: This will be used to gather all the research data

    **RESPONSE FORMAT:**
    Provide a comprehensive research report including:
    1. Executive summary of findings
    2. Detailed analysis of each major aspect
    3. Comparative insights where relevant
    4. Sources and references
    5. Conclusions and potential next steps
    """
    
    return system_prompt

# Create all agents with appropriate tools
approval_agent: Agent = Agent(
    name="QueryApprover", 
    instructions=query_approval_prompt, 
    model=llm_model,
    tools=[handoff_to_requirement_gathering]
)

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
    tools=[execute_research_plan, web_search]
)

async def process_research_query(user_context: ResearchContext, query: str) -> Tuple[bool, str]:
    """Process a research query through the full agent pipeline"""
    try:
        # Step 1: Approval check
        print("\n🔍 Checking query approval...")
        approval_output = await Runner.run(
            starting_agent=approval_agent, 
            input=query,
            context=user_context
        )
        
        # Parse approval response
        approval_data = json.loads(approval_output.final_output)
        is_approved = approval_data.get("approved", False)
        reason = approval_data.get("reason", "")
        action = approval_data.get("action", "reject")
        
        if not is_approved:
            return False, f"Query rejected: {reason}"
        
        print("✅ Query approved! Gathering requirements...")
        
        # Step 2: Requirement gathering via handoff
        handoff_result = await handoff_to_requirement_gathering(user_context, query)
        handoff_data = json.loads(handoff_result)
        
        if handoff_data.get("status") != "handoff_complete":
            return False, "Failed to gather requirements"
        
        questions = user_context.requirement_questions
        
        # If no questions needed, proceed directly to planning
        if not questions:
            print("✅ No additional requirements needed. Planning research...")
            planning_handoff = await handoff_to_planning(user_context, {})
            planning_data = json.loads(planning_handoff)
            
            if planning_data.get("status") == "planning_complete":
                print("✅ Research plan created. Executing...")
                final_result = await execute_research_plan(user_context)
                return True, final_result
            else:
                return False, "Research planning failed"
        
        # Step 3: Ask requirement questions
        print(f"\n🔍 To provide the best research, I need some details:")
        requirements = {}
        
        for i, question in enumerate(questions, 1):
            answer = input(f"\nQ{i}: {question}\nYour answer: ").strip()
            if answer:
                requirements[f"requirement_{i}"] = answer
        
        # Step 4: Handoff to planning with requirements
        print("✅ Requirements gathered. Planning research...")
        planning_handoff = await handoff_to_planning(user_context, requirements)
        planning_data = json.loads(planning_handoff)
        
        if planning_data.get("status") != "planning_complete":
            return False, "Research planning failed"
        
        # Step 5: Execute research plan
        print("✅ Research plan created. Executing...")
        final_result = await execute_research_plan(user_context)
        
        return True, final_result
        
    except Exception as e:
        return False, f"Error processing query: {e}"

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
            
            approved, result = await process_research_query(user_context, query)
            
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

asyncio.run(call_agent())