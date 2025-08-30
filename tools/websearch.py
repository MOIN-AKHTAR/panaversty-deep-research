import os
from agents import  RunContextWrapper, function_tool
from dataclass.research import ResearchContext
from tavily import AsyncTavilyClient

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
