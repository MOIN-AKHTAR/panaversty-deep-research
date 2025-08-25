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
from tavily import AsyncTavilyClient
from pydantic import BaseModel

# Import agent framework components
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, RunContextWrapper, handoff, RunResult, function_tool

# Mock EmptyInputError if not available
class EmptyInputError(Exception):
    pass

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
        
        # Format the search results
        if resp and 'results' in resp:
            formatted_results = []
            for i, result in enumerate(resp['results'][:top_k], 1):
                formatted_results.append(
                    f"{i}. {result.get('title', 'No title')}\n"
                    f"   URL: {result.get('url', 'No URL')}\n"
                    f"   Content: {result.get('content', 'No content')[:200]}...\n"
                )
            
            # Store results in context
            wrapper.context.research_results[query] = "\n".join(formatted_results)
            return "\n".join(formatted_results)
        return "No results found."
    except Exception as e:
        return f"Search error: {e}"


@function_tool
async def get_current_date_info(wrapper: RunContextWrapper[ResearchContext]) -> str:
    return json.dumps({
        "current_date": wrapper.context.current_date,
        "current_year": wrapper.context.current_year
    })

def citation_agent_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    system_prompt = f"""
    You are a Citation Agent. Your role is to ensure proper citation and referencing of all research sources.

    # CURRENT DATE CONTEXT
    Today's Date: {CURRENT_DATE}
    Current Year: {CURRENT_YEAR}

    **INSTRUCTIONS:**
    - Analyze the research data and identify all sources
    - Format citations properly using academic standards
    - Ensure all facts, statistics, and quotes are properly attributed
    - Include publication dates and timestamps where available
    - Note the currency of information (current as of {CURRENT_YEAR})
    - Create a comprehensive references section
    - Use consistent citation style throughout

    **RESPONSE FORMAT:**
    Return the research data with properly formatted citations and a references section.
    """
    
    return system_prompt

citation_agent = Agent(
    name="CitationAgent", 
    instructions=citation_agent_prompt, 
    model=llm_model
)




def reflection_agent_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    system_prompt = f"""
    You are a Reflection Agent. Your role is to provide critical analysis and quality assurance for research reports.

    # CURRENT DATE CONTEXT
    Today's Date: {CURRENT_DATE}
    Current Year: {CURRENT_YEAR}

    **INSTRUCTIONS:**
    - Analyze the research report for completeness and accuracy
    - Check if information is current and up-to-date (as of {CURRENT_YEAR})
    - Identify any gaps, biases, or missing perspectives
    - Check for logical consistency and coherence
    - Evaluate the strength of evidence and sources
    - Suggest areas for improvement or further research

    **RESPONSE FORMAT:**
    Provide a critical reflection including strengths, areas for improvement, and suggestions.
    """
    
    return system_prompt

reflection_agent = Agent(
    name="ReflectionAgent", 
    instructions=reflection_agent_prompt, 
    model=llm_model
)


# def research_agent_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
#     interests_list = ", ".join(context.context.profile.interests)
#     user_name = context.context.profile.name
#     research_plan = "\n".join([f"{i+1}. {query}" for i, query in enumerate(context.context.research_plan)]) if context.context.research_plan else "No plan provided"
    
#     system_prompt = f"""
#     You are a research execution agent for {user_name}. Your role is to conduct web research based on the provided research plan.

#     # CURRENT DATE CONTEXT
#     Today's Date: {CURRENT_DATE}
#     Current Year: {CURRENT_YEAR}

#     # RESEARCH AGENT PROFILE
#     User: {user_name}
#     Specialized Research Interests: {interests_list}
#     Results Count: {context.context.profile.results_count} per query

#     # RESEARCH PLAN TO EXECUTE:
#     {research_plan}

#     **INSTRUCTIONS:**
#     - Execute each search query from the research plan using the web_search tool
#     - For each query, call: web_search("your exact search query here")
#     - Focus on current and relevant information (current year: {CURRENT_YEAR})
#     - After completing all searches, analyze and compile a comprehensive research report
#     - Include key findings, insights, and relevant URLs
#     - Make the report well-structured and easy to read
#     - Store the final report in context.final_report

#     **RESPONSE FORMAT:**
#     Execute each search query systematically, then compile and return the final research report.
#     """
    
#     return system_prompt


def research_agent_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a research synthesis agent for {user_name}. Your role is to compile comprehensive research reports.

    # CURRENT DATE CONTEXT
    Today's Date: {CURRENT_DATE}
    Current Year: {CURRENT_YEAR}

    # RESEARCH AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **INSTRUCTIONS:**
    1. Execute the research plan using web_search tool
    2. Use get_current_date_info tool to ensure time accuracy
    3. Use CitationAgent tool to properly cite all sources
    4. Use RflectionAgent tool to analyze and improve the report
    5. Synthesize a comprehensive, well-structured final report

    **TOOLS AVAILABLE:**
    - web_search: Research each query in the plan
    - get_current_date_info: Get current date context
    - CitationAgent: Ensure proper citation of all sources
    - RflectionAgent: Get critical feedback on report quality

    **RESPONSE FORMAT:**
    Provide the final research report.
    """
    
    return system_prompt

research_agent = Agent(
    name="ResearchAgent", 
    instructions=research_agent_prompt, 
    model=llm_model,
    tools=[ 
        web_search, 
        citation_agent.as_tool(tool_name="CitationAgent",tool_description="Generates proper academic citations and ensures all research sources are properly attributed with consistent formatting."),
        reflection_agent.as_tool(tool_name="RflectionAgent", tool_description="Provides critical analysis of research quality, identifies gaps and biases, and suggests improvements for completeness and accuracy.")
        ],
    
)

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
    - Analyze the original query: "{context.context.approved_query}"
    - Create a set of 3-5 specific search queries that will comprehensively cover the topic
    - Focus on current and relevant information (current year: {CURRENT_YEAR})
    - Include time-specific queries when appropriate
    - Make queries specific and actionable for web search
    - Store the plan in context.research_plan as a list of strings
    - After creating the plan, use: handoff(research_agent)

    **RESPONSE FORMAT:**
    Create your research plan, then use: handoff(research_agent)
    """
    
    return system_prompt

planning_agent = Agent(
    name="ResearchPlanner", 
    instructions=planning_agent_prompt, 
    model=llm_model,
    handoffs=[handoff(research_agent)]
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
    
    **RESPONSE FORMAT:**
    Return JSON with approved status, query, rejection reason (if any), and final result.
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
    output: RunResult = await Runner.run(
        starting_agent=current_agent, 
        input=current_input,
        context=context,
        max_turns=50
    )
    
    print("======================> AGENT NAME <================")
    print(output._last_agent.name)
    
    if output._last_agent.name == "ResearchPlanner":
        print("======================> RESEARCH PLAN <================")    
        if context.research_plan:
            for i, query in enumerate(context.research_plan, 1):
                print(f"{i}. {query}")
        else:
            print("No research plan created")
    
    elif output._last_agent.name == "ResearchAgent":
        print("======================> RESEARCH RESULTS <================")
        if context.research_results:
            for query, result in context.research_results.items():
                print(f"🔍 Query: {query}")
                print(f"📊 Results preview: {result[:200]}...")
                print("-" * 50)
        else:
            print("No research results found")
        
        print("======================> FINAL REPORT <================")
        if context.final_report:
            print(context.final_report)
            return True, context.final_report
        else:
            print("No final report generated")
    
    print("======================> FINAL OUTPUT <================")
    print(output.final_output)    
    
    
    return False, "Agent chain completed"

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
            # Reset context for new query
            user_context.research_plan = []
            user_context.research_results = {}
            user_context.final_report = ""
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
        import traceback
        traceback.print_exc()
    finally:
        print("\nClosing Agent....")

if __name__ == "__main__":
    # Run the main agent
    asyncio.run(call_agent())