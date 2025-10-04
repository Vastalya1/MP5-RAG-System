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
        # Initialize the model - using the free tier model name
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
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
        """Process a query through the Gemini API.

        Args:
            query: The original user query string to be rewritten

        Returns:
            Optional[str]: The rewritten query with medical insurance terminology, or None if the operation fails
        """
        try:
            # Format the prompt with the user's query
            formatted_prompt = self.REWRITE_PROMPT.format(user_query=query)
            
            # Configure generation parameters
            generation_config = {
                "temperature": 0.3,  # Lower temperature for more focused output
                "top_p": 1,
                "top_k": 1,
                "max_output_tokens": 2048,
            }
            
            # Get response from Gemini
            response = await self.model.generate_content_async(
                formatted_prompt,
                generation_config=generation_config
            )
            
            if response and hasattr(response, 'text') and response.text:
                # Extract and return the rewritten query
                rewritten_query = response.text.strip()
                print(f"Original query: {query}")
                print(f"Rewritten query: {rewritten_query}")
                return rewritten_query
            else:
                print("Error: Invalid or empty response from Gemini API")
                print(f"Response object: {response}")
                return None
                
        except Exception as e:
            print(f"Error in query rewriting: {str(e)}")
            print(f"Error type: {type(e)}")
            return None
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
