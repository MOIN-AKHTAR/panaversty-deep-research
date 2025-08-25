"""
This is my deep research agent project
"""

import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Tuple
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
        "query": "the original query text"
    }}

    **EXAMPLES:**
    Query: "Explain attention mechanisms in AI?"
    → {{"approved": true, "reason": "AI is within user interests", "query": "Explain attention mechanisms in AI?"}}

    Query: "Tell me about quantum physics"
    → {{"approved": false, "reason": "Quantum physics not in user interests", "query": "Tell me about quantum physics"}}

    Query: "What is my name?"
    → {{"approved": false, "reason": "Meta-question about user profile", "query": "What is my name?"}}
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

    **INSTRUCTIONS:**
    - Use the web_search tool to gather current information
    - Provide detailed, well-structured answers with sources
    - Include relevant URLs from your research
    - Focus on accuracy and completeness
    
    **RESPONSE FORMAT:**
    Provide a comprehensive research report including:
    1. Clear answer to the query
    2. Detailed explanations with supporting evidence
    3. Sources and URLs from your research
    4. Any relevant context or additional information
    """
    
    return system_prompt

# Create both agents
approval_agent: Agent = Agent(
    name="QueryApprover", 
    instructions=query_approval_prompt, 
    model=llm_model
)

research_agent: Agent = Agent(
    name="ResearchSpecialist", 
    instructions=research_agent_prompt, 
    model=llm_model, 
    tools=[web_search]
)

async def process_query_with_approval(user_context: ResearchContext, query: str) -> Tuple[bool, str]:
    """Process query through approval agent first"""
    try:
        # Step 1: Get approval decision
        approval_output = await Runner.run(
            starting_agent=approval_agent, 
            input=query,
            context=user_context
        )
        
        # Parse the JSON response
        import json
        try:
            approval_data = json.loads(approval_output.final_output)
            is_approved = approval_data.get("approved", False)
            reason = approval_data.get("reason", "")
            
            if is_approved:
                # Step 2: If approved, pass to research agent
                user_context.approved_query = query
                user_context.is_approved = True
                
                research_output = await Runner.run(
                    starting_agent=research_agent, 
                    input=query,
                    context=user_context
                )
                
                return True, research_output.final_output
            else:
                return False, f"Query rejected: {reason}"
                
        except json.JSONDecodeError:
            return False, "Error: Approval agent response format invalid"
            
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
        
        # Create user context
        user_context = ResearchContext(profile=UserProfile(name=name, interests=interests))
        
        while True:
            print("\n" + "="*50)
            query = input("\nEnter your research query (or 'quit' to exit): ").strip()
            
            if query.lower() == 'quit':
                break
                
            if not query:
                print("Please enter a valid query.")
                continue
            
            print("\n🔍 Checking query approval...")
            approved, result = await process_query_with_approval(user_context, query)
            
            if approved:
                print(f"\n✅ APPROVED RESEARCH RESULTS:")
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