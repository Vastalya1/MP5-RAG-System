import google.generativeai as genai
from typing import Optional

class QueryRewriter:
    def __init__(self, api_key: str):
        """
        Initialize the Query Rewriter with Gemini API
        
        Args:
            api_key: Google API key for accessing Gemini
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # The prompt template for query rewriting
        self.REWRITE_PROMPT = """You are a query reformulation assistant specialized in the domain of medical insurance.

Your goal is to take a user's natural-language question and rewrite it into a clear,
factual, and context-rich query that will maximize the accuracy of information retrieval
from a database of medical insurance documents.

Follow these instructions carefully:

1. Preserve the original intent of the user's question.
2. Expand vague references or abbreviations into complete, explicit terms.
3. Replace general words with precise **medical or insurance-related terminology**
   whenever possible — use jargon that typically appears in policy documents.
   Examples:
      - "doctor fees" → "consultation charges"
      - "hospital stay" → "inpatient hospitalization"
      - "insurance amount" → "sum insured"
      - "things not covered" → "policy exclusions"
      - "time before coverage starts" → "waiting period"
      - "medicine costs" → "pharmaceutical expenses"
      - "room cost" → "room rent limit"
4. Avoid pronouns like "it" or "they" — make every reference explicit.
5. Do NOT answer the question; only rewrite it.
6. Keep the rewritten query in one complete sentence.
7. Ensure the rewritten query sounds like it was written for a legal or policy document.

User Query:
"{user_query}"

Rewritten, retrieval-optimized query with appropriate medical insurance terminology:
"""

    async def rewrite_query(self, query: str) -> Optional[str]:
        """
        Rewrite the user's query using the Gemini API
        
        Args:
            query: The original user query
            
        Returns:
            str: The rewritten query with medical insurance terminology
        """
        try:
            # Format the prompt with the user's query
            formatted_prompt = self.REWRITE_PROMPT.format(user_query=query)
            
            # Get response from Gemini
            response = await self.model.generate_content_async(formatted_prompt)
            
            # Extract and return the rewritten query
            rewritten_query = response.text.strip()
            print(f"Original query: {query}")
            print(f"Rewritten query: {rewritten_query}")
            
            return rewritten_query
            
        except Exception as e:
            print(f"Error in query rewriting: {str(e)}")
            return None

# # Example usage
# if __name__ == "__main__":
#     import asyncio
    
#     async def main():
#         API_KEY = "your-api-key-here"  # Replace with your actual API key
#         rewriter = QueryRewriter(API_KEY)
        
#         # Example query
#         test_query = "How much do I need to pay for a doctor's visit?"
#         rewritten = await rewriter.rewrite_query(test_query)
#         print(f"Rewritten query: {rewritten}")
    
#     asyncio.run(main())
