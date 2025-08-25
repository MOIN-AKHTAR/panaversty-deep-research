"""
This is my deep research agent project with improved context-aware approval
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

def is_related_to_interests(query: str, interests: List[str]) -> bool:
    """
    Intelligent function to determine if a query is related to user interests
    using both keyword matching and semantic understanding
    """
    query_lower = query.lower()
    interests_lower = [interest.lower() for interest in interests]
    
    # Common interest categories and their related topics
    interest_categories = {
        'wrestling': [
            'wwe', 'aew', 'royal rumble', 'wrestlemania', 'summerslam', 'raw', 'smackdown',
            'nxt', 'championship', 'belt', 'ppv', 'pay-per-view', 'wrestler', 'superstar',
            'tag team', 'title match', 'wrestling move', 'finisher', 'promo', 'storyline',
            'wrestling history', 'wrestling legends', 'wrestling events'
        ],
        'sports': [
            'football', 'basketball', 'soccer', 'baseball', 'hockey', 'tennis', 'golf',
            'cricket', 'rugby', 'olympics', 'championship', 'tournament', 'player', 'team',
            'score', 'game', 'match', 'sports history', 'sports statistics'
        ],
        'music': [
            'artist', 'band', 'album', 'song', 'genre', 'concert', 'tour', 'lyrics',
            'music production', 'music history', 'music theory', 'instrument', 'musician'
        ],
        'movies': [
            'film', 'actor', 'director', 'movie', 'cinema', 'hollywood', 'bollywood',
            'oscars', 'awards', 'movie review', 'film analysis', 'movie history'
        ],
        'technology': [
            'computer', 'software', 'hardware', 'programming', 'coding', 'ai', 'machine learning',
            'cloud', 'devops', 'cybersecurity', 'data science', 'web development', 'mobile apps'
        ],
        'ai': [
            'artificial intelligence', 'machine learning', 'deep learning', 'neural networks',
            'natural language processing', 'computer vision', 'ai ethics', 'llm', 'gpt',
            'transformer models', 'reinforcement learning'
        ],
        'coding': [
            'programming', 'developer', 'software', 'code', 'algorithm', 'data structure',
            'python', 'javascript', 'java', 'c++', 'web development', 'app development',
            'debugging', 'version control', 'git'
        ],
        'devops': [
            'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'ci/cd', 'continuous integration',
            'continuous deployment', 'infrastructure as code', 'cloud', 'aws', 'azure', 'gcp',
            'containerization', 'orchestration'
        ]
    }
    
    # First, check direct keyword matches
    for interest in interests_lower:
        if interest in query_lower:
            return True
            
        # Check if query contains any related topics for this interest
        if interest in interest_categories:
            for related_topic in interest_categories[interest]:
                if related_topic in query_lower:
                    return True
    
    # Check for broader semantic relationships
    broader_relationships = {
        'wrestling': ['sports entertainment', 'professional wrestling', 'wwe', 'aew'],
        'sports': ['athletics', 'competition', 'game', 'tournament'],
        'music': ['audio', 'sound', 'recording', 'performance'],
        'movies': ['cinema', 'films', 'entertainment', 'acting'],
        'technology': ['tech', 'innovation', 'digital', 'computer science'],
        'ai': ['artificial intelligence', 'machine intelligence', 'neural', 'algorithm'],
        'coding': ['programming', 'software development', 'scripting', 'developer'],
        'devops': ['development operations', 'software delivery', 'automation', 'infrastructure']
    }
    
    for interest in interests_lower:
        if interest in broader_relationships:
            for related_term in broader_relationships[interest]:
                if related_term in query_lower:
                    return True
    
    # If no matches found, be more permissive for general knowledge about interests
    for interest in interests_lower:
        if interest in ['wrestling', 'sports', 'music', 'movies']:
            # For entertainment topics, allow general questions about the field
            general_terms = ['what is', 'explain', 'tell me about', 'history of', 'best']
            for term in general_terms:
                if term in query_lower and any(word in query_lower for word in interest.split()):
                    return True
    
    return False

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
    3. Questions about sub-topics, events, or concepts within these interests
    4. Historical, technical, or conceptual questions about these interests
    5. Comparative questions between elements within these interests
    
    B. REJECT these types of questions:
    1. Questions COMPLETELY UNRELATED to: {interests_list}
    2. Personal or off-topic questions
    3. Questions about completely different domains
    4. Questions that violate content policies

    **SPECIFIC EXAMPLES:**
    - If interests include "wrestling": APPROVE "What is Royal Rumble?", "Wrestlemania history", "Best WWE matches"
    - If interests include "sports": APPROVE "Super Bowl winners", "Basketball rules", "Soccer strategies"
    - If interests include "music": APPROVE "Beatles discography", "Jazz history", "Music production techniques"
    - If interests include "movies": APPROVE "Oscar winners", "Film directing techniques", "Movie genre history"
    
    **HANDOFF DECISION:**
    - If query is APPROVED and needs clarification → use handoff to Requirement Gathering Agent
    - If query is APPROVED and clear → use handoff to Planning Agent
    - If query is REJECTED → respond with clear rejection explanation

    **RESPONSE FORMAT:**
    For approved queries: Use the handoff function to route to appropriate agent
    For rejected queries: Respond with JSON: {{"status": "rejected", "reason": "explanation"}}
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
    - After gathering requirements, use handoff to Planning Agent

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
    - After creating the plan, use handoff to Research Agent for execution

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

async def run_agent_chain(starting_agent: Agent, input_text: str, context: ResearchContext) -> Tuple[bool, str]:
    """Run the agent chain with proper handoff handling"""
    current_agent = starting_agent
    current_input = input_text
    max_iterations = 5
    
    for iteration in range(max_iterations):
        print(f"\n🤖 {current_agent.name} is processing your request...")
        
        # Run the current agent
        output = await Runner.run(
            starting_agent=current_agent, 
            input=current_input,
            context=context
        )
        
        # Check if the response is a rejection
        if current_agent == approval_router and is_rejection_response(output.final_output):
            reason = get_rejection_reason(output.final_output)
            return False, f"Query rejected: {reason}"
        
        # Check if this is the final output from research agent
        if current_agent == research_agent:
            return True, output.final_output
        
        # Check if we need to process requirement questions
        if (current_agent == requirement_gathering_agent):
            try:
                # Try to parse the JSON response to get questions
                data = json.loads(output.final_output)
                if 'questions' in data and data['questions']:
                    context.requirement_questions = data['questions']
                    
                    # Process user requirements
                    requirements = await process_user_requirements(context, context.requirement_questions)
                    context.requirements = requirements
                    
                    # Move to planning agent with the requirements
                    current_agent = planning_agent
                    current_input = json.dumps({
                        "original_query": context.approved_query,
                        "user_requirements": requirements
                    })
                    continue
            except:
                # If JSON parsing fails, continue with the output as is
                pass
        
        # Simple agent chain logic
        if current_agent == approval_router:
            # For vague queries, go to requirement gathering
            vague_indicators = ['research', 'study', 'learn about', 'tell me about', 'what about']
            if any(indicator in current_input.lower() for indicator in vague_indicators):
                current_agent = requirement_gathering_agent
            else:
                current_agent = planning_agent
        elif current_agent == requirement_gathering_agent:
            current_agent = planning_agent
        elif current_agent == planning_agent:
            current_agent = research_agent
        else:
            break
            
        current_input = output.final_output
    
    return False, "Maximum iterations reached without completing research"

async def call_agent():
    try:
        print("Wait loading agent for you ........")
        name = (input("Your name: ")).strip()
        if not name:
            raise EmptyInputError("Name cannot be empty")
        
        interests_input = (input("Your topic of interests you want to research separated by comma (e.g., Wrestling, AI, Music, Movies): ") or "AI").strip()
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

# Test function for wrestling interests
async def test_wrestling_queries():
    """Test function to demonstrate wrestling-related queries"""
    print("Testing wrestling-related queries...")
    
    # Create test context with wrestling interest
    test_context = ResearchContext(
        profile=UserProfile(name="WrestlingFan", interests=["Wrestling"], results_count=3)
    )
    
    # Test queries that should be APPROVED for wrestling interest
    test_queries = [
        "What is Royal Rumble?",
        "Tell me about Wrestlemania",
        "History of WWE",
        "Best wrestling matches of all time",
        "How does a wrestling storyline work?",
        "What are the different wrestling championships?",
        "Explain the rules of a ladder match",
        "Who are the greatest wrestlers in history?",
    ]
    
    for query in test_queries:
        print(f"\n--- Testing query: '{query}' ---")
        test_context.approved_query = query
        
        # First check with our intelligent function
        is_related = is_related_to_interests(query, test_context.profile.interests)
        print(f"Intelligent check: {'✅ RELATED' if is_related else '❌ UNRELATED'}")
        
        # Then test with the approval agent
        approved, result = await run_agent_chain(approval_router, query, test_context)
        
        if approved:
            print(f"✅ Agent approved: {result[:100]}..." if len(result) > 100 else f"✅ Agent approved: {result}")
        else:
            print(f"❌ Agent rejected: {result}")

if __name__ == "__main__":
    # Uncomment to test wrestling queries
    # asyncio.run(test_wrestling_queries())
    
    # Run the main agent
    asyncio.run(call_agent())