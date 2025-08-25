"""
This is my deep research agent project with requirement gathering
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

    **RESPONSE FORMAT:**
    You must respond with ONLY a JSON object in this exact format:
    {{
        "approved": true/false,
        "reason": "brief explanation for approval/rejection",
        "query": "the original query text",
        "needs_clarification": true/false  // Set to true if the query needs more details for effective research
    }}

    **EXAMPLES:**
    Query: "Explain attention mechanisms in AI?"
    → {{"approved": true, "reason": "AI is within user interests", "query": "Explain attention mechanisms in AI?", "needs_clarification": false}}

    Query: "Tell me about quantum physics"
    → {{"approved": false, "reason": "Quantum physics not in user interests", "query": "Tell me about quantum physics", "needs_clarification": false}}

    Query: "Research AI"
    → {{"approved": true, "reason": "AI is within user interests", "query": "Research AI", "needs_clarification": true}}
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

    Original Query: "Tell me about renewable energy"
    → {{
        "questions": [
            "Which type of renewable energy are you focusing on? (solar, wind, hydro, etc.)",
            "Are you interested in technical aspects, economic factors, or environmental impact?"
        ],
        "refined_query": "Research advancements in solar energy technology and its economic viability"
    }}
    """
    
    return system_prompt

def research_agent_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a research assistant for {user_name}. Your role is to provide comprehensive, well-researched answers to approved queries within these interest areas: {interests_list}.

    # RESEARCH AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **CONTEXT:**
    The user has provided these additional requirements: {context.context.requirements}

    **INSTRUCTIONS:**
    - Use the web_search tool to gather current information
    - Provide detailed, well-structured answers with sources
    - Include relevant URLs from your research
    - Focus on accuracy and completeness
    - Address all aspects mentioned in the user's requirements
    
    **RESPONSE FORMAT:**
    Provide a comprehensive research report including:
    1. Clear answer to the query
    2. Detailed explanations with supporting evidence
    3. Sources and URLs from your research
    4. Any relevant context or additional information
    """
    
    return system_prompt

# Create all three agents
approval_agent: Agent = Agent(
    name="QueryApprover", 
    instructions=query_approval_prompt, 
    model=llm_model
)

requirement_agent: Agent = Agent(
    name="RequirementGatherer", 
    instructions=requirement_gathering_prompt, 
    model=llm_model
)

research_agent: Agent = Agent(
    name="ResearchSpecialist", 
    instructions=research_agent_prompt, 
    model=llm_model, 
    tools=[web_search]
)

async def gather_requirements(user_context: ResearchContext, query: str) -> Tuple[Dict, str]:
    """Gather requirements for a research query"""
    try:
        # Get requirement questions
        requirement_output = await Runner.run(
            starting_agent=requirement_agent, 
            input=query,
            context=user_context
        )
        
        # Parse the JSON response
        requirement_data = json.loads(requirement_output.final_output)
        questions = requirement_data.get("questions", [])
        refined_query = requirement_data.get("refined_query", query)
        
        # Store questions in context
        user_context.requirement_questions = questions
        
        # If no questions needed, return empty requirements
        if not questions:
            return {}, refined_query
        
        # Ask questions and collect answers
        requirements = {}
        print(f"\n🔍 To provide the best research, I need some details:")
        
        for i, question in enumerate(questions, 1):
            answer = input(f"\nQ{i}: {question}\nYour answer: ").strip()
            if answer:
                requirements[f"q{i}"] = answer
        
        return requirements, refined_query
        
    except Exception as e:
        print(f"Error gathering requirements: {e}")
        return {}, query

async def process_query(user_context: ResearchContext, query: str) -> Tuple[bool, str]:
    """Process query through the full agent pipeline"""
    try:
        # Step 1: Get approval decision
        print("\n🔍 Checking query approval...")
        approval_output = await Runner.run(
            starting_agent=approval_agent, 
            input=query,
            context=user_context
        )
        
        # Parse the JSON response
        approval_data = json.loads(approval_output.final_output)
        is_approved = approval_data.get("approved", False)
        reason = approval_data.get("reason", "")
        needs_clarification = approval_data.get("needs_clarification", False)
        
        if not is_approved:
            return False, f"Query rejected: {reason}"
        
        # Step 2: If approved and needs clarification, gather requirements
        research_query = query
        if needs_clarification:
            requirements, research_query = await gather_requirements(user_context, query)
            user_context.requirements = requirements
            user_context.approved_query = research_query
        
        # Step 3: Pass to research agent
        user_context.is_approved = True
        print("\n🔬 Conducting research...")
        
        research_output = await Runner.run(
            starting_agent=research_agent, 
            input=research_query,
            context=user_context
        )
        
        return True, research_output.final_output
            
    except json.JSONDecodeError:
        return False, "Error: Agent response format invalid"
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
            results_count = int(input("How many search results would you like? (1-10, default 3): ") or "3")
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
            
            approved, result = await process_query(user_context, query)
            
            if approved:
                print(f"\n✅ RESEARCH RESULTS:")
                print(f"{result}")
            else:
                print(f"\n❌ {result}")
                
    except EmptyInputError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("\nClosing Agent....")

asyncio.run(call_agent())