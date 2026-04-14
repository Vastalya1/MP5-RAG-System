"""
Query Classifier using a small LLM to determine routing.

This classifier determines whether a query:
1. Requires RAG processing (medical/insurance document lookup)
2. Can be answered directly by LLM (general knowledge, greetings, etc.)
"""

from typing import Literal
from mistralai import Mistral


class QueryClassifier:
    """
    Classifies user queries to determine the appropriate processing path.
    Uses Mistral's lightweight model for fast classification.
    """
    
    ROUTE_RAG = "rag"
    ROUTE_DIRECT = "direct"
    
    def __init__(self, api_key: str):
        """
        Initialize the Query Classifier with Mistral API.
        
        Args:
            api_key: Mistral API key
        """
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-tiny"  # Fastest model for quick classification
        
        self.CLASSIFICATION_PROMPT = """You are a query router for a medical insurance policy assistant.

Your task is to classify whether a user query requires looking up information from medical insurance policy documents or can be answered directly.

CLASSIFY AS "rag" if the query:
- Asks about specific policy coverage, benefits, or exclusions
- Inquires about claim procedures or documentation
- Questions about premium amounts, deductibles, or co-pays
- Asks about waiting periods or policy terms
- Seeks information about specific medical procedures coverage
- Questions about network hospitals or providers
- Asks about policy renewal, cancellation, or portability
- Inquires about pre-existing conditions
- Any query that would need verification from policy documents

CLASSIFY AS "direct" if the query:
- Is a greeting (hello, hi, good morning)
- Is a general knowledge question not about insurance specifics
- Asks about what the assistant can do
- Is casual conversation or chitchat
- Asks for explanations of general medical/insurance concepts that don't require policy document lookup
- Thanks or acknowledgments
- Asks to repeat or clarify previous responses

Respond with ONLY one word: either "rag" or "direct"
Do not add any explanation, punctuation, or additional text."""

    def classify(self, query: str) -> Literal["rag", "direct"]:
        """
        Classify a query to determine the routing path.
        
        Args:
            query: The user's query string
            
        Returns:
            "rag" if the query needs document retrieval
            "direct" if the query can be answered directly by LLM
        """
        try:
            messages = [
                {"role": "system", "content": self.CLASSIFICATION_PROMPT},
                {"role": "user", "content": f"Query: {query}"},
            ]
            
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=0.0,  # Deterministic for classification
                max_tokens=10     # Only need one word
            )
            
            if response and response.choices:
                classification = response.choices[0].message.content.strip().lower()
                
                # Normalize response
                if "rag" in classification:
                    print(f"[Classifier] Query routed to: RAG")
                    return self.ROUTE_RAG
                elif "direct" in classification:
                    print(f"[Classifier] Query routed to: DIRECT LLM")
                    return self.ROUTE_DIRECT
                else:
                    # Default to RAG for safety (better to search than miss info)
                    print(f"[Classifier] Uncertain classification '{classification}', defaulting to RAG")
                    return self.ROUTE_RAG
            else:
                print("[Classifier] Empty response, defaulting to RAG")
                return self.ROUTE_RAG
                
        except Exception as e:
            print(f"[Classifier] Error in classification: {str(e)}, defaulting to RAG")
            return self.ROUTE_RAG

    async def classify_async(self, query: str) -> Literal["rag", "direct"]:
        """
        Async version of classify for use in async contexts.
        
        Args:
            query: The user's query string
            
        Returns:
            "rag" if the query needs document retrieval
            "direct" if the query can be answered directly by LLM
        """
        # Mistral's sync client works in async contexts
        # For true async, consider using httpx or aiohttp
        return self.classify(query)
