"""
This is my deep research agent project with current date/time awareness
"""

import os
import asyncio
import json
import re
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from inputError import EmptyInputError
from tavily import AsyncTavilyClient
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, RunContextWrapper, handoff

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
    cited_results: Dict = field(default_factory=dict)
    final_report: str = ""
    current_date: str = CURRENT_DATE
    current_year: str = CURRENT_YEAR

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
async def add_citations(wrapper: RunContextWrapper[ResearchContext], research_data: str) -> str:
    """
    📚 Use Citation Agent to properly format and add citations to research data.
    Ensures all sources are properly referenced and credited.
    """
    try:
        print("📚 Adding citations to research data...")
        
        # Call citation agent
        citation_output = await Runner.run(
            starting_agent=citation_agent, 
            input=research_data,
            context=wrapper.context
        )
        
        # Store cited results in context
        wrapper.context.cited_results = citation_output.final_output
        
        return citation_output.final_output
    except Exception as e:
        return f"Citation error: {e}"

@function_tool
async def reflect_on_research(wrapper: RunContextWrapper[ResearchContext], research_report: str) -> str:
    """
    🤔 Use Reflection Agent to analyze and improve the research report.
    Provides critical analysis, identifies gaps, and suggests improvements.
    """
    try:
        print("🤔 Reflecting on research quality...")
        
        # Call reflection agent
        reflection_output = await Runner.run(
            starting_agent=reflection_agent, 
            input=research_report,
            context=wrapper.context
        )
        
        return reflection_output.final_output
    except Exception as e:
        return f"Reflection error: {e}"

@function_tool
async def get_current_date_info(wrapper: RunContextWrapper[ResearchContext]) -> str:
    """
    📅 Get current date and time information for accurate research context.
    """
    return json.dumps({
        "current_date": CURRENT_DATE,
        "current_year": CURRENT_YEAR,
        "current_month": CURRENT_MONTH,
        "timestamp": datetime.now().isoformat()
    })

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

    **HANDOFF DECISION:**
    - If query is RELATED but VAGUE → use: handoff(requirement_gathering_agent)
    - If query is RELATED and CLEAR → use: handoff(planning_agent)
    - If query is UNRELATED → respond with JSON rejection

    **RESPONSE FORMAT:**
    For APPROVED queries: USE THE EXACT HANDOFF FUNCTION
    Example for vague query: handoff(requirement_gathering_agent)
    Example for clear query: handoff(planning_agent)
    
    For REJECTED queries: {{"status": "rejected", "reason": "explanation"}}

    **TIME AWARENESS:**
    - Remember that current year is {CURRENT_YEAR}
    - Ensure research focuses on current and relevant information
    """
    
    return system_prompt

def requirement_gathering_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a requirement gathering agent for {user_name}. Your role is to ask clarifying questions for vague research queries.

    # CURRENT DATE CONTEXT
    Today's Date: {CURRENT_DATE}
    Current Year: {CURRENT_YEAR}

    # REQUIREMENT GATHERING AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **INSTRUCTIONS:**
    - You receive queries that are approved but need clarification
    - Ask 2-3 concise, relevant questions to gather missing details
    - Focus on aspects like: specificity, time frame, format, criteria, scope
    - Consider current date context: {CURRENT_YEAR}
    - Store questions in context.requirement_questions
    - After asking questions, use: handoff(planning_agent)

    **TIME-RELATED CLARIFICATIONS:**
    - Ask about time frames if not specified (current, historical, specific years)
    - Clarify if user wants latest information or historical perspective
    - Suggest current year {CURRENT_YEAR} as default for recent information

    **RESPONSE FORMAT:**
    Ask your clarifying questions, then use: handoff(planning_agent)
    """
    
    return system_prompt

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

    **TIME-AWARE QUERY PLANNING:**
    - Include current year {CURRENT_YEAR} in queries for recent information
    - Consider including time frames for historical context
    - Ensure queries are timely and relevant

    **RESPONSE FORMAT:**
    Create your research plan, then use: handoff(research_agent)
    """
    
    return system_prompt

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

    **CITATION FORMAT:**
    - For web sources: [Title](URL) - Author, Publication Date
    - For recent information: Note "Current as of {CURRENT_YEAR}"
    - Include timestamps for time-sensitive information
    - Highlight most recent sources

    **RESPONSE FORMAT:**
    Return the research data with properly formatted citations and a references section.
    """
    
    return system_prompt

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
    - Assess the timeliness and relevance of information

    **TIME-RELATED REFLECTION:**
    - Is the information current and relevant to {CURRENT_YEAR}?
    - Are sources recent and authoritative?
    - Does the report acknowledge the publication dates of sources?
    - Is historical context properly framed against current information?

    **RESPONSE FORMAT:**
    Provide a critical reflection including strengths, areas for improvement, and suggestions.
    """
    
    return system_prompt

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
    3. Use add_citations tool to properly cite all sources
    4. Use reflect_on_research tool to analyze and improve the report
    5. Synthesize a comprehensive, well-structured final report
    6. Ensure information is current and relevant to {CURRENT_YEAR}

    **TOOLS AVAILABLE:**
    - web_search: Research each query in the plan
    - get_current_date_info: Get current date context
    - add_citations: Ensure proper citation of all sources
    - reflect_on_research: Get critical feedback on report quality

    **TIME-AWARE RESEARCH:**
    - Focus on current information (as of {CURRENT_YEAR})
    - Include publication dates in your analysis
    - Note the currency and relevance of each source
    - Prioritize recent information while providing historical context

    **FINAL REPORT STRUCTURE:**
    1. Executive Summary (current as of {CURRENT_YEAR})
    2. Detailed Analysis with timestamps
    3. Key Findings with relevance to current context
    4. Supporting Evidence (with proper citations and dates)
    5. Conclusions and Current Recommendations
    6. References Section with publication dates
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

citation_agent = Agent(
    name="CitationSpecialist",
    instructions=citation_agent_prompt,
    model=llm_model
)

reflection_agent = Agent(
    name="QualityReflector",
    instructions=reflection_agent_prompt,
    model=llm_model
)

research_agent = Agent(
    name="ResearchSynthesizer", 
    instructions=research_agent_prompt, 
    model=llm_model,
    tools=[web_search, get_current_date_info, add_citations, reflect_on_research]
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

def is_handoff_response(response: str) -> bool:
    """Check if the response contains a handoff"""
    return "handoff(" in response and ")" in response

def extract_handoff_agent(response: str) -> Optional[str]:
    """Extract the agent name from handoff response"""
    match = re.search(r"handoff\((\w+)\)", response)
    if match:
        return match.group(1)
    return None

async def process_user_requirements(user_context: ResearchContext, agent_output: str) -> Dict:
    """Process user responses to requirement questions"""
    print(f"\n🔍 {agent_output}")
    
    requirements = {}
    questions = []
    
    # Extract questions from the agent output (before any handoff)
    clean_output = agent_output.split('handoff(')[0].strip()
    lines = clean_output.split('\n')
    
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
            context.final_report = output.final_output
            return True, output.final_output
        
        # Check if there's a handoff to the next agent
        if hasattr(output, 'next_agent') and output.next_agent:
            print(f"🔄 Handoff detected from {current_agent.name} to {output.next_agent.name}")
            current_agent = output.next_agent
            current_input = output.final_output
            continue
        
        # Check if the response contains a handoff function call
        if is_handoff_response(output.final_output):
            handoff_agent_name = extract_handoff_agent(output.final_output)
            if handoff_agent_name:
                print(f"🔄 Handoff function detected to: {handoff_agent_name}")
                
                # Find the target agent
                target_agent = None
                for agent in [requirement_gathering_agent, planning_agent, research_agent, citation_agent, reflection_agent]:
                    if agent.name.lower() == handoff_agent_name.lower():
                        target_agent = agent
                        break
                
                if target_agent:
                    current_agent = target_agent
                    # Use the content before the handoff as input
                    clean_input = output.final_output.split('handoff(')[0].strip()
                    current_input = clean_input if clean_input else "Proceed with the research"
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