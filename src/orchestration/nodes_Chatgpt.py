"""
LangGraph Node Implementations for the RAG Orchestration System.

This module contains the three main processing nodes:
1. Direct LLM Node - Handles queries that don't need document lookup
2. RAG Process Node - Handles queries requiring document retrieval
3. Web Scraping Node - Handles low similarity score scenarios (placeholder)
"""

from typing import Any, Dict, Optional
import sys
from pathlib import Path

from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parent.parent))

from output.answerGeneration_Chatgpt import AnswerGenerator
from queryDecomposition.orchestrator_Chatgpt import QueryDecompositionOrchestrator
from queryRewriter.rewriting_Chatgpt import QueryRewriter
from retriever.reranking_Chatgpt import ChunkReranker
from retriever.retrival import retrivalModel
from tavily_fallback.tavily_client import TavilySearchClient
from tavily_fallback.tavily_service import TavilyService
from .rag_pipeline_Chatgpt import RAGSubQueryProcessor


class DirectLLMNode:
    """
    Node for handling queries that don't require document lookup.
    Provides direct conversational responses using the LLM.
    """

    def __init__(self, api_key: str):
        """
        Initialize the Direct LLM Node.

        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

        self.SYSTEM_PROMPT = """You are a helpful medical insurance assistant. 
You are currently responding to a general query that doesn't require looking up specific policy documents.

Guidelines:
- Be friendly and conversational
- For general insurance concepts, provide clear explanations
- If asked about specific policy details, politely indicate that you'd need the user to ask about their specific policy
- Keep responses concise but helpful
- Don't make up specific numbers, coverage amounts, or policy details
- If the query seems to actually need policy document lookup, suggest rephrasing the question to get specific policy information
- IMPORTANT: Return PLAIN TEXT only. Do not use any markdown formatting (no asterisks, no bold, no bullet points with dashes). Just use plain sentences and paragraphs."""

    async def process(self, query: str) -> Dict[str, Any]:
        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )

            if response and response.choices:
                answer = (response.choices[0].message.content or "").strip()
                return {
                    "answer": answer,
                    "justification": None,
                    "sources": [],
                    "route_taken": "direct_llm",
                    "success": True,
                }
            return {
                "answer": "I apologize, but I couldn't generate a response. Please try again.",
                "justification": None,
                "sources": [],
                "route_taken": "direct_llm",
                "success": False,
            }

        except Exception as e:
            print(f"[DirectLLMNode] Error: {str(e)}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "justification": None,
                "sources": [],
                "route_taken": "direct_llm",
                "success": False,
                "error": str(e),
            }


class RAGProcessNode:
    """
    Node for handling queries that require document retrieval and RAG processing.
    Uses the full RAG pipeline: Query Rewriting -> Retrieval -> Reranking -> Answer Generation
    """

    def __init__(
        self,
        api_key: str,
        rewriter: Optional[QueryRewriter] = None,
        retriever: Optional[retrivalModel] = None,
        reranker: Optional[ChunkReranker] = None,
        answer_generator: Optional[AnswerGenerator] = None,
    ):
        self.api_key = api_key
        self.pipeline = RAGSubQueryProcessor(
            api_key=api_key,
            rewriter=rewriter,
            retriever=retriever,
            reranker=reranker,
            answer_generator=answer_generator,
        )
        self.decomposition_orchestrator = QueryDecompositionOrchestrator(
            api_key=api_key,
            rag_processor=self.pipeline,
        )

    async def process(
        self,
        query: str,
        scope: str = "shared",
        username: Optional[str] = None,
        collection_name: Optional[str] = None,
        document_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            print("[RAGProcessNode] Starting decomposition-aware RAG pipeline...")
            result = await self.decomposition_orchestrator.process_query(
                query=query,
                scope=scope,
                username=username,
                collection_name=collection_name,
                document_filter=document_filter,
            )
            sub_query_results = result.get("sub_query_results", [])
            top_distances = [
                item.get("top_chunk_distance")
                for item in sub_query_results
                if item.get("top_chunk_distance") is not None
            ]
            return {
                "answer": result.get("answer", ""),
                "justification": result.get("justification"),
                "sources": result.get("sources", []),
                "route_taken": "rag",
                "rewritten_query": result.get("rewritten_query"),
                "success": result.get("success", False),
                "needs_web_scraping": result.get("needs_web_scraping", False),
                "top_chunk_distance": min(top_distances) if top_distances else None,
                "retrieval_debug": result.get("retrieval_debug"),
                "sub_query_results": sub_query_results,
            }

        except Exception as e:
            print(f"[RAGProcessNode] Error: {str(e)}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "justification": None,
                "sources": [],
                "route_taken": "rag",
                "success": False,
                "error": str(e),
                "needs_web_scraping": False,
            }


class WebScrapingNode:
    """
    Node for handling low-confidence RAG results using Tavily web search.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        if api_key:
            self.client = OpenAI(api_key=api_key)
        try:
            self.service = TavilyService(TavilySearchClient())
            self.tavily_available = True
            print("[WebScrapingNode] Tavily service initialized")
        except Exception as e:
            print(f"[WebScrapingNode] Tavily not available: {e}")
            self.tavily_available = False

    async def process(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        if not self.tavily_available:
            return {
                "answer": "Web search is currently unavailable. Please try again or contact support.",
                "justification": "Tavily service unavailable",
                "sources": [],
                "route_taken": "web_scraping",
                "success": False,
                "error": "Tavily API not configured",
            }

        try:
            print(f"[WebScrapingNode] Searching web for: {query}")
            result = self.service.get_answer(query)

            return {
                "answer": result.get("answer", ""),
                "justification": "Answer generated using web search",
                "sources": result.get("sources", []),
                "route_taken": "web_scraping",
                "success": True,
            }

        except Exception as e:
            print(f"[WebScrapingNode] Error: {str(e)}")
            return {
                "answer": "Unable to fetch information from web search. Please try again.",
                "justification": None,
                "sources": [],
                "route_taken": "web_scraping",
                "success": False,
                "error": str(e),
            }


async def direct_llm_node(state: Dict, api_key: str) -> Dict:
    node = DirectLLMNode(api_key)
    result = await node.process(state["query"])
    return {**state, "result": result}


async def rag_process_node(
    state: Dict,
    api_key: str,
    rewriter: Optional[QueryRewriter] = None,
    retriever: Optional[retrivalModel] = None,
    reranker: Optional[ChunkReranker] = None,
    answer_generator: Optional[AnswerGenerator] = None,
) -> Dict:
    node = RAGProcessNode(
        api_key,
        rewriter=rewriter,
        retriever=retriever,
        reranker=reranker,
        answer_generator=answer_generator,
    )
    result = await node.process(
        state["query"],
        scope=state.get("scope", "shared"),
        username=state.get("username"),
        collection_name=state.get("collection_name"),
        document_filter=state.get("document_filter"),
    )
    return {**state, "result": result}


async def web_scraping_node(state: Dict, api_key: str) -> Dict:
    node = WebScrapingNode(api_key)
    result = await node.process(state["query"], context=state.get("result"))
    return {**state, "result": result}
