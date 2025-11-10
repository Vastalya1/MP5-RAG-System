from typing import List, Dict, Optional
import re

from src.retriever.retrival import retrivalModel
from src.retriever.reranking_mistral import ChunkReranker
from src.output.answerGeneration_mistral import AnswerGenerator


class SimplePolicyClassifier:
    """Very small classifier based on keywords/regex to detect medical-policy queries.

    It's intentionally lightweight so it can run offline and deterministically. For
    production you'd replace this with a trained classifier or a call to a small LLM.
    """

    POLICY_KEYWORDS = [
        r"policy", r"coverage", r"covered", r"claim", r"pre-?existing",
        r"deductible", r"premium", r"inpatient", r"outpatient",
        r"prescription", r"copay", r"co-?pay", r"eligib", r"network",
        r"referral", r"prior[- ]authorization", r"benefit"
    ]

    def __init__(self, threshold: int = 1):
        # threshold = number of keyword matches to consider as policy-related
        self.threshold = threshold
        self.patterns = [re.compile(pat, re.I) for pat in self.POLICY_KEYWORDS]

    def is_policy_query(self, text: str) -> bool:
        if not text:
            return False
        score = 0
        for pat in self.patterns:
            if pat.search(text):
                score += 1
                if score >= self.threshold:
                    return True
        return False


class Router:
    """Dispatch queries either to the retrieval-based RAG pipeline or to the LLM-only path.

    Contract:
      - input: raw user query string
      - output: dict (same shape as previous answer response) or error dict
    """

    def __init__(self, mistral_api_key: str):
        self.classifier = SimplePolicyClassifier()
        # instantiate pipeline components used for retrieval path
        self.retriever = retrivalModel()
        self.reranker = ChunkReranker(mistral_api_key)
        self.answer_generator = AnswerGenerator(mistral_api_key)

    async def dispatch(self, original_query: str) -> Dict:
        """Decide route and execute pipeline.

        If classifier thinks it's a policy question -> do retrieval pipeline.
        Otherwise -> call LLM-only answer generator with no chunks (it will answer general queries).
        """
        try:
            is_policy = self.classifier.is_policy_query(original_query)

            # For policy queries: full RAG path (rewrite -> retrieve -> rerank -> answer)
            if is_policy:
                # Rewriting is done elsewhere in the app; here we keep the same contract and
                # expect the caller to provide a rewritten query. To remain compatible with
                # existing app, we'll simply return a flag indicating this route.
                return {"route": "retrieval", "query": original_query}

            # Non-policy: direct LLM route (no retrieval). We'll ask the LLM to answer directly.
            else:
                # Use answer_generator but with empty chunks list; it will produce a best-effort reply.
                resp = await self.answer_generator.generate_answer(original_query, [])
                return {"route": "llm", "response": resp}

        except Exception as e:
            return {"error": str(e)}
