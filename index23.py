"""
This is my deep research agent project with intelligent requirement gathering
"""
import asyncio
import json
from datetime import datetime
from typing import Tuple
from dotenv import load_dotenv, find_dotenv

_: bool = load_dotenv(find_dotenv())


from inputError import EmptyInputError
from dataclass.research import ResearchContext
from dataclass.userprofile import UserProfile
from myagents.approval import approval_router
from agents import Agent, Runner,set_tracing_disabled,RunResult



# Get current date and time
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")
CURRENT_YEAR = datetime.now().strftime("%Y")


# Enable tracing to see handoffs
set_tracing_disabled(disabled=False)


def is_rejection_response(response: str) -> bool:
    """Check if the response indicates a rejection"""
    try:
        data = json.loads(response)
        return data.get("status") == "rejected"
    except:
        return False


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

        if output._last_agent.name=="ResearchAgent":
            print("======================> FINAL OUTPUT <================")
            return True, output.final_output
    
    
    return False, "Agent chain completed"

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
                # print(f"\n✅ FINAL RESEARCH REPORT (Current as of {CURRENT_YEAR}):")
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