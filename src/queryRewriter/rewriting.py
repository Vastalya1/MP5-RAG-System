from typing import Optional
from mistralai import Mistral
# from mistralai.models.chat_completion import ChatMessage

class QueryRewriter:
    def __init__(self, api_key: str):
        """
        Initialize the Query Rewriter with Mistral API
        
        Args:
            api_key: Mistral API key
        """
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-small"  # Fastest model in Mistral's lineup
        
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

    async def rewrite_query(self, query: str) -> Optional[str]:
        """Process a query through the Mistral API.

        Args:
            query: The original user query string to be rewritten

        Returns:
            Optional[str]: The rewritten query with medical insurance terminology, or None if the operation fails
        """
        try:
            # Create chat messages
            messages=[
                    {"role": "system", "content": self.REWRITE_PROMPT},
                    {"role": "user", "content": query},
                ]
            
            # Get response from Mistral API
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=0.3,
                top_p=0.95,
                max_tokens=150
            )
            
            if response and response.choices:
                # Extract the rewritten query
                rewritten_query = response.choices[0].message.content.strip()
                print(f"Original query: {query}")
                print(f"Rewritten query: {rewritten_query}")
                return rewritten_query
            else:
                print("Error: Empty response from Mistral API")
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
