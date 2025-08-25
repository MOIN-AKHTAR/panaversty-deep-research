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
    # New: Track requirement conversation
    requirement_conversation: List[Dict] = field(default_factory=list)

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

    **IMPORTANT:**
    - DO NOT explain your decision, just use the handoff function
    - DO NOT add any additional text before or after the handoff
    """
    
    return system_prompt

def requirement_gathering_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    original_query = context.context.approved_query
    
    # Build conversation history context
    conversation_history = ""
    if context.context.requirement_conversation:
        conversation_history = "\n# CONVERSATION HISTORY:\n"
        for i, conv in enumerate(context.context.requirement_conversation):
            conversation_history += f"Turn {i+1}:\n- User: {conv['user']}\n- Agent: {conv['agent']}\n\n"
    
    system_prompt = f"""
    You are an intelligent requirement gathering agent for {user_name}. Your role is to ask SMART, context-aware follow-up questions.

    # ORIGINAL USER QUERY: "{original_query}"
    # USER INTERESTS: {interests_list}
    # CURRENT DATE: {CURRENT_YEAR}

    {conversation_history}

    **SMART QUESTIONING STRATEGY:**
    - NEVER ask "what are you interested in?" - you already know their interests
    - NEVER ask generic questions - be specific and context-aware
    - ALWAYS remember the original query and build upon it
    - Ask questions that actually help narrow down the research scope
    - Build upon previous answers in the conversation

    **CONTEXT-AWARE QUESTION EXAMPLES:**

    Original: "greatest movies"
    → SMART: "Which criteria for 'greatest'? Critics' choice (Oscars), box office success, or cultural impact?"
    → SMART: "Any specific genre? All-time classics or recent decade?"
    → SMART: "Looking for a ranked list or detailed analysis of top films?"

    Original: "best cricket players"  
    → SMART: "Which format? Test cricket, ODI, or T20 specialists?"
    → SMART: "Current players or all-time legends? Any specific era?"
    → SMART: "Focus on batsmen, bowlers, or all-rounders?"

    Original: "Research AI"
    → SMART: "Which AI domain? Machine learning, neural networks, or practical applications?"
    → SMART: "Technical depth: beginner overview or research paper level?"
    → SMART: "Focus on recent advancements (2023-2024) or foundational concepts?"

    Original: "DevOps tools"
    → SMART: "Which area? CI/CD pipelines, containerization, or cloud infrastructure?"
    → SMART: "Specific platforms? AWS, Azure, Kubernetes, or general tools?"
    → SMART: "Looking for comparison, tutorials, or best practices?"

    **FOLLOW-UP FLOW:**
    1. Understand the original query context
    2. Ask 1-2 specific, intelligent questions that actually help
    3. Build upon previous answers - don't repeat questions
    4. When you have enough context, use: handoff(planning_agent)

    **BAD QUESTIONS TO AVOID:**
    ❌ "What are you interested in?"
    ❌ "Tell me more" 
    ❌ "What do you want to know?"
    ❌ Generic, context-less questions
    ❌ Questions that ignore previous conversation

    **RESPONSE FORMAT:**
    Ask intelligent, specific questions based on the context, then handoff when ready.
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

    **TOOLS AVAILABLE:**
    - web_search: Research each query in the plan
    - get_current_date_info: Get current date context
    - add_citations: Ensure proper citation of all sources
    - reflect_on_research: Get critical feedback on report quality

    **RESPONSE FORMAT:**
    Provide the final research report.
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

def find_agent_by_name(agent_name: str):
    """Find agent by name from available agents"""
    agents = {
        "requirementgatherer": requirement_gathering_agent,
        "researchplanner": planning_agent,
        "researchsynthesizer": research_agent,
        "citationspecialist": citation_agent,
        "qualityreflector": reflection_agent
    }
    return agents.get(agent_name.lower(), planning_agent)

async def run_agent_chain(starting_agent: Agent, input_text: str, context: ResearchContext) -> Tuple[bool, str]:
    """Run the agent chain with proper conversational requirement gathering"""
    current_agent = starting_agent
    current_input = input_text
    max_iterations = 20
    
    for iteration in range(max_iterations):
        print(f"\n🌀 [{iteration+1}] Current agent: {current_agent.name}")
        
        # Run the current agent
        output = await Runner.run(
            starting_agent=current_agent, 
            input=current_input,
            context=context
        )

        print("CONTEXT ------------->")
        print(context.research_plan)
        
        print(f"📤 Output: {output.final_output[:100]}{'...' if len(output.final_output) > 100 else ''}")
        
        # Check if the response is a rejection (from approval router)
        if current_agent == approval_router and is_rejection_response(output.final_output):
            reason = get_rejection_reason(output.final_output)
            return False, f"Query rejected: {reason}"
        
        # Check if this is the final output from research agent
        if current_agent == research_agent:
            context.final_report = output.final_output
            return True, output.final_output
        
        # Handle requirement gathering conversation
        if current_agent == requirement_gathering_agent:
            # Add to conversation history
            context.requirement_conversation.append({
                "user": current_input,
                "agent": output.final_output
            })
            
            # Check if agent is handing off
            if is_handoff_response(output.final_output):
                handoff_agent_name = extract_handoff_agent(output.final_output)
                if handoff_agent_name and "planning" in handoff_agent_name.lower():
                    print("✅ Requirements gathered! Moving to planning...")
                    current_agent = planning_agent
                    current_input = json.dumps({
                        "original_query": context.approved_query,
                        "requirements": extract_requirements_from_conversation(context.requirement_conversation),
                        "conversation_summary": f"User provided details through {len(context.requirement_conversation)} conversation turns"
                    })
                    continue
            
            # If not handing off, ask user for response
            print("\n💬 Requirement gathering conversation:")
            print("=" * 60)
            print(f"🤖 {output.final_output}")
            print("=" * 60)
            
            user_response = input("\n👤 Your response (or 'quit' to cancel): ").strip()
            if user_response.lower() == 'quit':
                return False, "Conversation cancelled by user"
                
            if not user_response:
                user_response = "I'm not sure, can you ask differently?"
                
            current_input = user_response
            continue  # Stay with requirement agent for next iteration
        
        # Handle other agents normally
        if hasattr(output, 'next_agent') and output.next_agent:
            current_agent = output.next_agent
            current_input = output.final_output
            continue
        
        if is_handoff_response(output.final_output):
            handoff_agent_name = extract_handoff_agent(output.final_output)
            if handoff_agent_name:
                current_agent = find_agent_by_name(handoff_agent_name)
                current_input = output.final_output.split('handoff(')[0].strip() or "Proceed with research"
                continue
        
        # Default progression
        if current_agent == approval_router:
            current_agent = requirement_gathering_agent if is_vague_query(current_input) else planning_agent
        elif current_agent == planning_agent:
            current_agent = research_agent
        else:
            current_agent = planning_agent  # Default fallback
        
        current_input = output.final_output
        
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