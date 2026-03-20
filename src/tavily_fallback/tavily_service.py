class TavilyService:
    def __init__(self, client):
        self.client = client

    def get_answer(self, query: str) -> dict:
        response = self.client.search(query)

        return {
            "answer": response.get("answer", ""),
            "sources": [
                {
                    "url": r.get("url"),
                    "title": r.get("title", "")
                }
                for r in response.get("results", [])
            ]
        }
