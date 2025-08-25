"""
This is my deep research agent project with intelligent requirement gathering
"""

import os
import asyncio
import json
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv, find_dotenv
from inputError import EmptyInputError
from pydantic import BaseModel
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, RunContextWrapper, handoff, RunResult


_: bool = load_dotenv(find_dotenv())

gemini_api_key: str | None = os.getenv("OPENAI_API_KEY")

# Enable tracing to see handoffs
set_tracing_disabled(disabled=False)

# 1. Which LLM Service?
external_client: AsyncOpenAI = AsyncOpenAI(api_key=gemini_api_key)

# 2. Which LLM Model?
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=external_client)

# Get current date and time
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")
CURRENT_YEAR = datetime.now().strftime("%Y")
CURRENT_MONTH = datetime.now().strftime("%B %Y")


# Define your data structure
class ApproveAgentResponse(BaseModel):
    approved: bool
    query: str
    rejection_reason: str
    final_result: str

@dataclass
class UserProfile:
    name: str = "Moin"
    interests: List[str] = field(default_factory=lambda: [])
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
    cited_results: Dict = field(default_factory=dict)
    final_report: str = ""
    current_date: str = CURRENT_DATE
    current_year: str = CURRENT_YEAR
    # New: Track requirement conversation
    requirement_conversation: List[Dict] = field(default_factory=list)


def planning_agent_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a research planning agent for {user_name}. Your role is to create a comprehensive research plan.

    # CURRENT DATE CONTEXT
    Today's Date: {CURRENT_DATE}
    Current Year: {CURRENT_YEAR}

    # PLANNING AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **INSTRUCTIONS:**
    - Analyze the original query and any user requirements
    - Create a set of specific search queries that will comprehensively cover the topic
    - Focus on current and relevant information (current year: {CURRENT_YEAR})
    - Include time-specific queries when appropriate
    - Plan for 3-5 focused search queries maximum
    - Store the plan in context.research_plan
    - After creating the plan, use: handoff(research_agent)

    **RESPONSE FORMAT:**
    Create your research plan, then use: handoff(research_agent)
    """
    
    return system_prompt


planning_agent = Agent(
    name="ResearchPlanner", 
    instructions=planning_agent_prompt, 
    model=llm_model
)

def query_approval_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a query approval agent for {user_name}. Your role is to determine if research queries are related to these specific interest areas: {interests_list}.

    # CURRENT DATE CONTEXT
    Today's Date: {CURRENT_DATE}
    Current Year: {CURRENT_YEAR}

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
    """
    
    return system_prompt

approval_router = Agent(
    name="ApprovalRouter",
    instructions=query_approval_prompt,
    model=llm_model,
    output_type=ApproveAgentResponse,
    handoffs=[handoff(planning_agent)]
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

def is_handoff_response(response: str) -> bool:
    """Check if the response contains a handoff"""
    return "handoff(" in response and ")" in response

def extract_handoff_agent(response: str) -> Optional[str]:
    """Extract the agent name from handoff response"""
    match = re.search(r"handoff\((\w+)\)", response)
    if match:
        return match.group(1)
    return None

def is_vague_query(query: str) -> bool:
    """Determine if a query is vague and needs clarification"""
    query_lower = query.lower()
    vague_patterns = [
        r'who is (?:the|a) top',
        r'best (?:player|artist|movie|song)',
        r'greatest (?:player|artist|movie|song)',
        r'who are (?:the|some) best',
        r'research',
        r'study',
        r'learn about',
        r'tell me about',
        r'what about',
        r'information on',
        r'overview of',
    ]
    
    for pattern in vague_patterns:
        if re.search(pattern, query_lower):
            return True
    
    return len(query.split()) <= 3

def extract_requirements_from_conversation(conversation: List[Dict]) -> Dict:
    """Extract structured requirements from conversation history"""
    requirements = {}
    for i, msg in enumerate(conversation):
        if msg.get('user') and not msg['user'].startswith('{'):
            requirements[f"requirement_{i+1}"] = msg['user']
    return requirements

async def run_agent_chain(starting_agent: Agent, input_text: str, context: ResearchContext) -> Tuple[bool, str]:
    """Run the agent chain with proper conversational requirement gathering"""
    current_agent = starting_agent
    current_input = input_text
     # Run the current agent
    output:RunResult = await Runner.run(
            starting_agent=current_agent, 
            input=current_input,
            context=context
        )
    print("======================> AGENT NAME <================")
    print(output._last_agent.name)
    print("======================> AGENT PLAN <================")    
    print(context.research_plan)
    print("======================> FINAL OUTPUT <================")
    print(output.final_output)    
    return False, "Agent chain completed without final output"

async def call_agent():
    try:
        print("Wait loading agent for you ........")
        print(f"📅 Current Date: {CURRENT_DATE}")
        print(f"📅 Current Year: {CURRENT_YEAR}")
        
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
        
        # Create user context with current date info
        user_context = ResearchContext(
            profile=UserProfile(name=name, interests=interests, results_count=results_count),
            current_date=CURRENT_DATE,
            current_year=CURRENT_YEAR
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
            # Reset conversation for new query
            user_context.requirement_conversation = []
            
            # Run the agent chain
            approved, result = await run_agent_chain(approval_router, query, user_context)
            
            if approved:
                print(f"\n✅ FINAL RESEARCH REPORT (Current as of {CURRENT_YEAR}):")
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

if __name__ == "__main__":
    # Run the main agent
    asyncio.run(call_agent())