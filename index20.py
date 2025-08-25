"""
This is my deep research agent project with intelligent requirement gathering
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
            return "\n".join(formatted_results)
        return "No results found."
    except Exception as e:
        return f"Search error: {e}"

@function_tool
async def add_citations(wrapper: RunContextWrapper[ResearchContext], research_data: str) -> str:
    """
    📚 Format and add citations to research data properly.
    """
    try:
        print("📚 Adding citations to research data...")
        
        # Simple citation formatting - you can enhance this
        citation_prompt = f"""
        Please format the following research data with proper citations and references.
        Include source URLs and ensure academic-style formatting.
        
        Research Data:
        {research_data}
        
        Format the output with proper in-text citations and a references section.
        """
        
        citation_result = await llm_model.run(
            messages=[{"role": "user", "content": citation_prompt}]
        )
        
        wrapper.context.cited_results = citation_result
        return citation_result
    except Exception as e:
        return f"Citation error: {e}"

@function_tool
async def reflect_on_research(wrapper: RunContextWrapper[ResearchContext], research_report: str) -> str:
    """
    🤔 Analyze and improve the research report quality.
    """
    try:
        print("🤔 Reflecting on research quality...")
        
        reflection_prompt = f"""
        Analyze this research report critically:
        
        {research_report}
        
        Provide:
        1. Strengths of the report
        2. Areas for improvement
        3. Missing information or gaps
        4. Suggestions for enhancement
        5. Overall quality assessment
        
        Current Year: {CURRENT_YEAR}
        """
        
        reflection_result = await llm_model.run(
            messages=[{"role": "user", "content": reflection_prompt}]
        )
        
        return reflection_result
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
    
    return f"""
    You are a query approval agent for {user_name}. Determine if research queries are related to: {interests_list}.

    CURRENT DATE: {CURRENT_DATE}
    USER INTERESTS: {interests_list}

    **DECISION RULES:**
    ✅ APPROVE if related to: {interests_list}
    ❌ REJECT if completely unrelated

    **HANDOFF LOGIC:**
    - If RELATED but VAGUE → handoff(requirement_gathering_agent)
    - If RELATED and CLEAR → handoff(research_agent)
    - If UNRELATED → respond with JSON: {{"status": "rejected", "reason": "explanation"}}

    **RESPONSE FORMAT:**
    Only use handoff() function or JSON rejection. No additional text.
    """

def requirement_gathering_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    original_query = context.context.approved_query
    
    conversation_history = ""
    if context.context.requirement_conversation:
        conversation_history = "\n# CONVERSATION HISTORY:\n"
        for i, conv in enumerate(context.context.requirement_conversation[-3:]):  # Last 3 turns
            conversation_history += f"Turn {i+1}:\n- User: {conv['user']}\n- Agent: {conv['agent']}\n\n"
    
    return f"""
    You are an intelligent requirement gathering agent. Ask SMART, specific questions.

    ORIGINAL QUERY: "{original_query}"
    USER INTERESTS: {interests_list}
    CURRENT YEAR: {CURRENT_YEAR}

    {conversation_history}

    **SMART QUESTIONING:**
    - Ask specific, context-aware questions
    - Build upon previous answers
    - NEVER ask generic "what are you interested in?"
    - Focus on narrowing research scope

    **EXAMPLES:**
    Query: "greatest movies" → "Which criteria? Critics' choice, box office, or cultural impact?"
    Query: "AI research" → "Which AI domain? ML, neural networks, or applications?"

    When you have enough context, use: handoff(research_agent)
    """

def research_agent_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    
    return f"""
    You are a research synthesis agent. Create comprehensive research reports.

    CURRENT DATE: {CURRENT_DATE}
    USER INTERESTS: {interests_list}

    **INSTRUCTIONS:**
    1. Analyze the research query and requirements
    2. Create 3-5 focused search queries using web_search tool
    3. Use get_current_date_info for time accuracy
    4. Synthesize results into a comprehensive report
    5. Use add_citations for proper referencing
    6. Use reflect_on_research for quality assurance

    **RESEARCH STRATEGY:**
    - Focus on current information ({CURRENT_YEAR})
    - Include multiple perspectives
    - Verify facts across sources
    - Structure report logically

    Provide the final research report with proper formatting.
    """

def planning_agent_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    """Planning agent for creating research strategies"""
    return f"""
    You are a research planning agent. Create strategic search queries.

    CURRENT YEAR: {CURRENT_YEAR}
    QUERY: {context.context.approved_query}

    Create 3-5 focused search queries that will comprehensively cover the topic.
    Consider time relevance, multiple perspectives, and depth of coverage.

    Store the plan and use: handoff(research_agent)
    """

# Create all agent instances
requirement_gathering_agent = Agent(
    name="RequirementGatherer", 
    instructions=requirement_gathering_prompt, 
    model=llm_model
)

research_agent = Agent(
    name="ResearchSynthesizer", 
    instructions=research_agent_prompt, 
    model=llm_model,
    tools=[web_search, get_current_date_info, add_citations, reflect_on_research]
)

planning_agent = Agent(
    name="ResearchPlanner", 
    instructions=planning_agent_prompt, 
    model=llm_model
)

approval_router = Agent(
    name="ApprovalRouter",
    instructions=query_approval_prompt,
    model=llm_model
)

# Set up handoffs
requirement_gathering_agent.handoffs = [handoff(research_agent)]
planning_agent.handoffs = [handoff(research_agent)]
approval_router.handoffs = [handoff(requirement_gathering_agent), handoff(research_agent)]

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

def is_related_to_interests(query: str, interests: List[str]) -> bool:
    """Check if query is related to user interests"""
    query_lower = query.lower()
    for interest in interests:
        if interest.lower() in query_lower:
            return True
    return False

async def autonomous_agent_chain(starting_agent: Agent, input_text: str, context: ResearchContext) -> Tuple[bool, str]:
    """Autonomous agent chain with proper handoffs"""
    current_agent = starting_agent
    current_input = input_text
    max_iterations = 10
    
    for iteration in range(max_iterations):
        print(f"\n🌀 [{iteration+1}] Agent: {current_agent.name}")
        
        # Run the current agent
        output = await Runner.run(
            starting_agent=current_agent, 
            input=current_input,
            context=context
        )
        
        print(f"📤 Output: {output.final_output[:150]}{'...' if len(output.final_output) > 150 else ''}")
        
        # Handle rejection from approval router
        if current_agent == approval_router and is_rejection_response(output.final_output):
            return False, get_rejection_reason(output.final_output)
        
        # Handle final output from research agent
        if current_agent == research_agent:
            context.final_report = output.final_output
            return True, output.final_output
        
        # Handle requirement gathering conversation
        if current_agent == requirement_gathering_agent:
            context.requirement_conversation.append({
                "user": current_input,
                "agent": output.final_output
            })
            
            if is_handoff_response(output.final_output):
                # Extract next agent from handoff
                next_agent_name = extract_handoff_agent(output.final_output)
                if next_agent_name:
                    next_agent = globals().get(f"{next_agent_name.lower()}_agent")
                    if next_agent:
                        current_agent = next_agent
                        current_input = "Proceed with research based on gathered requirements"
                        continue
            
            # If not handing off, continue conversation
            print(f"\n💬 {output.final_output}")
            user_response = input("\n👤 Your response: ").strip()
            if user_response.lower() == 'quit':
                return False, "Conversation cancelled"
                
            current_input = user_response
            continue
        
        # Handle handoffs for other agents
        if is_handoff_response(output.final_output):
            next_agent_name = extract_handoff_agent(output.final_output)
            if next_agent_name:
                next_agent = globals().get(f"{next_agent_name.lower()}_agent")
                if next_agent:
                    current_agent = next_agent
                    current_input = output.final_output.split('handoff(')[0].strip() or "Proceed"
                    continue
        
        # Default progression
        if current_agent == approval_router:
            # Auto-detect if query needs requirement gathering
            if is_vague_query(current_input):
                current_agent = requirement_gathering_agent
            else:
                current_agent = research_agent
            current_input = output.final_output
        else:
            # Default to research agent
            current_agent = research_agent
            current_input = output.final_output
    
    return False, "Maximum iterations reached"

async def call_agent():
    try:
        print("🔍 Research Agent Initializing...")
        print(f"📅 Current Date: {CURRENT_DATE}")
        
        name = input("Your name: ").strip() or "User"
        interests_input = input("Your interests (comma-separated): ").strip() or "AI, Technology"
        interests = [interest.strip() for interest in interests_input.split(",")]
        
        try:
            results_count = int(input("Results per query (1-10): ") or "3")
            results_count = max(1, min(10, results_count))
        except ValueError:
            results_count = 3
        
        user_context = ResearchContext(
            profile=UserProfile(name=name, interests=interests, results_count=results_count)
        )
        
        while True:
            print("\n" + "="*50)
            query = input("\nResearch query (or 'quit'): ").strip()
            
            if query.lower() == 'quit':
                break
                
            if not query:
                continue
            
            print(f"\n🔍 Processing: {query}")
            user_context.approved_query = query
            user_context.requirement_conversation = []
            
            # Check if query is related to interests first
            if not is_related_to_interests(query, interests):
                print(f"❌ Query not related to your interests: {', '.join(interests)}")
                continue
            
            approved, result = await autonomous_agent_chain(approval_router, query, user_context)
            
            if approved:
                print(f"\n✅ FINAL RESEARCH REPORT:")
                print("="*60)
                print(result)
                print("="*60)
            else:
                print(f"\n❌ {result}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("\nClosing Agent...")

if __name__ == "__main__":
    asyncio.run(call_agent())