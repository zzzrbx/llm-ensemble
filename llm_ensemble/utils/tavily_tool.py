import os
from langchain_core.tools import tool
from tavily import TavilyClient


@tool
def search_the_web(query: str):
    """
    Search the internet for current events, news, and real-time information.
    Use this for any questions about the world that require up-to-date data.
    """
    # Initialize Tavily Client (lazy initialization to ensure env vars are loaded)
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    # Execute the search using the raw SDK
    response = tavily.search(
        query=query,
        max_results=10,
        search_depth="basic"
    )

    # Format the results into a clean string for the LLM
    # This prevents the LLM from getting confused by raw JSON metadata
    formatted_results = []
    for res in response['results']:
        formatted_results.append(
            f"Title: {res['title']}\nURL: {res['url']}\nContent: {res['content']}\n"
        )

    return "\n---\n".join(formatted_results)
