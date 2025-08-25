"""
This is my deep research agent project with Citation and Reflection agents
"""

import os
import asyncio
import json
import re
from dotenv import load_dotenv, find_dotenv
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from inputError import EmptyInputError
from tavily import AsyncTavilyClient
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, RunContextWrapper, handoff

_: bool = load_dotenv(find_dotenv())

gemini_api_key: str | None = os.getenv("OPENAI_API_KEY")

# Tracing disabled
set_tracing_disabled(disabled=False)

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
    cited_results: Dict = field(default_factory=dict)
    final_report: str = ""

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

def query_approval_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a query approval agent for {user_name}. Your role is to determine if research queries are related to these specific interest areas: {interests_list}.

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
    - If query is RELATED but VAGUE → use handoff to Requirement Gathering Agent
    - If query is RELATED and CLEAR → use handoff to Planning Agent
    - If query is UNRELATED → respond with rejection explanation

    **RESPONSE FORMAT:**
    For APPROVED queries: USE THE HANDOFF FUNCTION to route to the appropriate agent
    For REJECTED queries: Respond with JSON: {{"status": "rejected", "reason": "explanation"}}
    """
    
    return system_prompt

def requirement_gathering_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    interests_list = ", ".join(context.context.profile.interests)
    user_name = context.context.profile.name
    
    system_prompt = f"""
    You are a requirement gathering agent for {user_name}. Your role is to ask clarifying questions for vague research queries.

    # REQUIREMENT GATHERING AGENT PROFILE
    User: {user_name}
    Specialized Research Interests: {interests_list}

    **INSTRUCTIONS:**
    - You receive queries that are approved but need clarification
    - Ask 2-3 concise, relevant questions to gather missing details
    - Focus on aspects like: specificity, time frame, format, criteria, scope
    - Store questions in context.requirement_questions
    - After asking questions, use handoff to Planning Agent

    **RESPONSE FORMAT:**
    First ask your clarifying questions in natural language, then use handoff to planning_agent.
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
    - Analyze the original query and any user requirements
    - Create a set of specific search queries that will comprehensively cover the topic
    - Plan for 3-5 focused search queries maximum
    - Store the plan in context.research_plan
    - After creating the plan, use handoff to Research Agent for execution

    **RESPONSE FORMAT:**
    Create your research plan, then use handoff to research_agent.
    """
    
    return system_prompt

def citation_agent_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    system_prompt = """
    You are a Citation Agent. Your role is to ensure proper citation and referencing of all research sources.

    **INSTRUCTIONS:**
    - Analyze the research data and identify all sources
    - Format citations properly using academic standards
    - Ensure all facts, statistics, and quotes are properly attributed
    - Create a comprehensive references section
    - Use consistent citation style throughout

    **CITATION FORMAT:**
    - For web sources: [Title](URL) - Author, Date
    - For academic sources: Author (Year). Title. Journal.
    - Include timestamps for time-sensitive information

    **RESPONSE FORMAT:**
    Return the research data with properly formatted citations and a references section.
    """
    
    return system_prompt

def reflection_agent_prompt(context: RunContextWrapper[ResearchContext], agent: Agent[ResearchContext]) -> str:
    system_prompt = """
    You are a Reflection Agent. Your role is to provide critical analysis and quality assurance for research reports.

    **INSTRUCTIONS:**
    - Analyze the research report for completeness and accuracy
    - Identify any gaps, biases, or missing perspectives
    - Check for logical consistency and coherence
    - Evaluate the strength of evidence and sources
    - Suggest areas for improvement or further research
    - Provide a quality assessment score (1-10)

    **REFLECTION AREAS:**
    1. **Completeness**: Does it cover all aspects of the query?
    2. **Accuracy**: Are facts verified and sources credible?
    3. **Balance**: Are multiple perspectives considered?
    4. **Depth**: Is the analysis sufficiently detailed?
    5. **Clarity**: Is the report well-organized and understandable?

    **RESPONSE FORMAT:**
    Provide a critical reflection including:
    - Strengths of the research
    - Areas for improvement
    - Quality assessment score
    - Suggestions for enhancement
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
    1. Execute the research plan using web_search tool
    2. Use add_citations tool to properly cite all sources
    3. Use reflect_on_research tool to analyze and improve the report
    4. Synthesize a comprehensive, well-structured final report
    5. Include proper context, analysis, and recommendations

    **TOOLS AVAILABLE:**
    - web_search: Research each query in the plan
    - add_citations: Ensure proper citation of all sources
    - reflect_on_research: Get critical feedback on report quality

    **RESEARCH PROCESS:**
    For each search query in the plan:
    - Use web_search to gather information
    - Extract key insights and data
    - Ensure proper source tracking

    Then:
    - Use add_citations to format all references
    - Use reflect_on_research to get quality feedback
    - Incorporate feedback into final report

    **FINAL REPORT STRUCTURE:**
    1. Executive Summary
    2. Detailed Analysis
    3. Key Findings
    4. Supporting Evidence (with proper citations)
    5. Conclusions and Recommendations
    6. References Section
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
    tools=[web_search, add_citations, reflect_on_research]
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

async def process_user_requirements(user_context: ResearchContext, agent_output: str) -> Dict:
    """Process user responses to requirement questions"""
    print(f"\n🔍 {agent_output}")
    
    requirements = {}
    questions = []
    
    # Extract questions from the agent output
    lines = agent_output.split('\n')
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
    max_iterations = 12
    
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
            
            # Store the original query
            user_context.approved_query = query
            
            # Run the agent chain
            approved, result = await run_agent_chain(approval_router, query, user_context)
            
            if approved:
                print(f"\n✅ FINAL RESEARCH REPORT:")
                print("="*60)
                print(f"{result}")
                print("="*60)
                
                # Show additional context information
                if hasattr(user_context, 'cited_results') and user_context.cited_results:
                    print(f"\n📊 Research included properly cited sources")
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