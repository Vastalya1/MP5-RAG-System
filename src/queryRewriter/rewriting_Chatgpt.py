from typing import Optional

from openai import OpenAI


class QueryRewriter:
    def __init__(self, api_key: str):
        """
        Initialize the Query Rewriter with OpenAI API

        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

        # The prompt template for query rewriting
        self.REWRITE_PROMPT = """You are a specialized medical insurance query reformulation expert. Your task is to rewrite user questions into clear, factual queries using proper insurance terminology.

Follow these rules:
1. Use precise medical/insurance terms:
   - Replace "doctor fees" with "consultation charges"
   - Replace "hospital stay" with "inpatient hospitalization"
   - Replace "insurance amount" with "sum insured"
   - Replace "things not covered" with "policy exclusions"
   - Replace "time before coverage starts" with "waiting period"
   - Replace "medicine costs" with "pharmaceutical expenses"
   - Replace "room cost" with "room rent limit"
2. Keep original question intent
3. Use formal policy document language
4. Return only one sentence
5. Avoid pronouns - be explicit

Respond only with the rewritten query, no additional text or explanations."""

    def rewrite_query_sync(self, query: str) -> Optional[str]:
        """Synchronously rewrite a query using medical insurance terminology."""
        try:
            messages = [
                {"role": "system", "content": self.REWRITE_PROMPT},
                {"role": "user", "content": query},
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                top_p=0.95,
                max_tokens=150,
            )

            if response and response.choices:
                rewritten_query = (response.choices[0].message.content or "").strip()
                print(f"Original query: {query}")
                print(f"Rewritten query: {rewritten_query}")
                return rewritten_query or None

            print("Error: Empty response from OpenAI API")
            return None

        except Exception as e:
            print(f"Error in query rewriting: {str(e)}")
            print(f"Error type: {type(e)}")
            return None

    async def rewrite_query(self, query: str) -> Optional[str]:
        """Process a query through the OpenAI API.

        Args:
            query: The original user query string to be rewritten

        Returns:
            Optional[str]: The rewritten query with medical insurance terminology, or None if the operation fails
        """
        return self.rewrite_query_sync(query)
