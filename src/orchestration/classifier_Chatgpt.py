"""
Query Classifier using a small LLM to determine routing.

This classifier determines whether a query:
1. Requires RAG processing (medical/insurance document lookup)
2. Can be answered directly by LLM (general knowledge, greetings, etc.)
"""

from typing import Literal

from openai import OpenAI
from shared.logging_utils import get_logger, log_error, log_info


logger = get_logger(__name__)


class QueryClassifier:
    """
    Classifies user queries to determine the appropriate processing path.
    Uses OpenAI's lightweight model for fast classification.
    """

    ROUTE_RAG = "rag"
    ROUTE_DIRECT = "direct"

    def __init__(self, api_key: str):
        """
        Initialize the Query Classifier with OpenAI API.

        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

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

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=10,
            )

            if response and response.choices:
                classification = (response.choices[0].message.content or "").strip().lower()

                if "rag" in classification:
                    log_info(logger, "query_classified", route=self.ROUTE_RAG)
                    return self.ROUTE_RAG
                if "direct" in classification:
                    log_info(logger, "query_classified", route=self.ROUTE_DIRECT)
                    return self.ROUTE_DIRECT

                log_error(logger, "query_classification_uncertain", classification=classification, fallback=self.ROUTE_RAG)
                return self.ROUTE_RAG

            log_error(logger, "query_classification_empty_response", fallback=self.ROUTE_RAG)
            return self.ROUTE_RAG

        except Exception as e:
            log_error(logger, "query_classification_failed", error_type=type(e).__name__, error=str(e), fallback=self.ROUTE_RAG)
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
        return self.classify(query)
