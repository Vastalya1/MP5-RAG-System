from tavily import TavilyClient
import os

class TavilySearchClient:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set")

        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str) -> dict:
        return self.client.search(
            query=query,
            search_depth="advanced",
            include_answer=True,
            max_results=5
        )
