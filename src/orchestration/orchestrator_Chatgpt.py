"""
LangGraph Orchestrator for RAG System

This module implements the main orchestration logic using LangGraph.
It routes queries between Direct LLM and RAG processing paths based on
classification performed BEFORE query rewriting.
"""

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from .classifier_Chatgpt import QueryClassifier
from .nodes_Chatgpt import DirectLLMNode, RAGProcessNode, WebScrapingNode
from shared.logging_utils import get_logger, log_error, log_info

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from output.answerGeneration_Chatgpt import AnswerGenerator
from queryRewriter.rewriting_Chatgpt import QueryRewriter
from retriever.reranking_Chatgpt import ChunkReranker
from retriever.retrival import retrivalModel


logger = get_logger(__name__)


class GraphState(TypedDict):
    query: str
    rewritten_query: Optional[str]
    scope: str
    username: Optional[str]
    collection_name: Optional[str]
    document_filter: Optional[str]
    route: Optional[Literal["rag", "direct"]]
    answer: Optional[str]
    justification: Optional[str]
    sources: List[Dict]
    success: bool
    route_taken: Optional[str]
    needs_web_scraping: bool
    error: Optional[str]
    sub_query_results: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class QueryOrchestrator:
    def __init__(
        self,
        api_key: str,
        rewriter: Optional[QueryRewriter] = None,
        retriever: Optional[retrivalModel] = None,
        reranker: Optional[ChunkReranker] = None,
        answer_generator: Optional[AnswerGenerator] = None,
    ):
        self.api_key = api_key
        self.classifier = QueryClassifier(api_key)
        self.direct_llm_node = DirectLLMNode(api_key)
        self.rag_node = RAGProcessNode(
            api_key,
            rewriter=rewriter,
            retriever=retriever,
            reranker=reranker,
            answer_generator=answer_generator,
        )
        self.web_scraping_node = WebScrapingNode(api_key)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(GraphState)

        builder.add_node("classifier_node", self._classifier_node)
        builder.add_node("direct_llm_node", self._direct_llm_node)
        builder.add_node("rag_node", self._rag_node)
        builder.add_node("web_scraping_node", self._web_scraping_node)

        builder.add_edge(START, "classifier_node")
        builder.add_conditional_edges(
            "classifier_node",
            self._route_query,
            {"direct": "direct_llm_node", "rag": "rag_node"},
        )
        builder.add_conditional_edges(
            "rag_node",
            self._check_similarity_threshold,
            {"proceed": END, "web_scrape": "web_scraping_node"},
        )
        builder.add_edge("web_scraping_node", END)
        builder.add_edge("direct_llm_node", END)

        compiled_graph = builder.compile()
        log_info(logger, "langgraph_compiled")
        return compiled_graph

    def _classifier_node(self, state: GraphState) -> dict:
        query = state["query"]
        log_info(logger, "orchestrator_classification_started", query_length=len(query))
        route = self.classifier.classify(query)
        return {
            "route": route,
            "metadata": {**state.get("metadata", {}), "classified_route": route},
        }

    async def _direct_llm_node(self, state: GraphState) -> dict:
        query = state["query"]
        log_info(logger, "orchestrator_direct_llm_started", query_length=len(query))
        result = await self.direct_llm_node.process(query)
        return {
            "answer": result.get("answer"),
            "justification": result.get("justification"),
            "sources": result.get("sources", []),
            "success": result.get("success", False),
            "route_taken": "direct_llm",
            "needs_web_scraping": False,
            "error": result.get("error"),
        }

    async def _rag_node(self, state: GraphState) -> dict:
        query = state["query"]
        scope = state.get("scope", "shared")
        username = state.get("username")
        collection_name = state.get("collection_name")
        document_filter = state.get("document_filter")
        log_info(logger, "orchestrator_rag_started", scope=scope, collection_name=collection_name, document_filter=document_filter)
        result = await self.rag_node.process(
            query=query,
            scope=scope,
            username=username,
            collection_name=collection_name,
            document_filter=document_filter,
        )
        return {
            "answer": result.get("answer"),
            "justification": result.get("justification"),
            "sources": result.get("sources", []),
            "success": result.get("success", False),
            "route_taken": "rag",
            "rewritten_query": result.get("rewritten_query"),
            "needs_web_scraping": result.get("needs_web_scraping", False),
            "retrieval_debug": result.get("retrieval_debug"),
            "sub_query_results": result.get("sub_query_results", []),
            "error": result.get("error"),
            "metadata": {
                **state.get("metadata", {}),
                "top_chunk_distance": result.get("top_chunk_distance"),
                "sub_query_results": result.get("sub_query_results", []),
            },
        }

    async def _web_scraping_node(self, state: GraphState) -> dict:
        metadata = state.get("metadata", {})
        rewritten_query = (
            state["query"]
            if metadata.get("decomposition_used")
            else (state.get("rewritten_query") or state["query"])
        )

        log_info(logger, "orchestrator_web_fallback_started")
        result = await self.web_scraping_node.process(query=rewritten_query, context=state)
        return {
            "answer": result.get("answer"),
            "justification": result.get("justification"),
            "sources": result.get("sources", []),
            "success": result.get("success", False),
            "route_taken": "tavily_web_search",
            "error": result.get("error"),
        }

    def _route_query(self, state: GraphState) -> Literal["direct", "rag"]:
        route = state.get("route", "rag")
        log_info(logger, "orchestrator_route_selected", route=route)
        return route

    def _check_similarity_threshold(self, state: GraphState) -> Literal["proceed", "web_scrape"]:
        if state.get("needs_web_scraping", False):
            return "web_scrape"
        return "proceed"

    async def process_query(
        self,
        query: str,
        scope: str = "shared",
        username: Optional[str] = None,
        collection_name: Optional[str] = None,
        document_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        initial_state: GraphState = {
            "query": query,
            "rewritten_query": None,
            "scope": scope,
            "username": username,
            "collection_name": collection_name,
            "document_filter": document_filter,
            "route": None,
            "answer": None,
            "justification": None,
            "sources": [],
            "success": False,
            "route_taken": None,
            "needs_web_scraping": False,
            "error": None,
            "sub_query_results": [],
            "metadata": {},
        }

        try:
            log_info(logger, "orchestrator_processing_started", scope=scope, collection_name=collection_name, document_filter=document_filter)
            result = await self.graph.ainvoke(initial_state)

            log_info(logger, "orchestrator_processing_completed", route_taken=result.get("route_taken"))
            return {
                "response": result.get("answer", ""),
                "justification": result.get("justification"),
                "sources": result.get("sources", []),
                "route_taken": result.get("route_taken"),
                "rewritten_query": result.get("rewritten_query"),
                "retrieval_debug": result.get("retrieval_debug"),
                "success": result.get("success", False),
                "needs_web_scraping": result.get("needs_web_scraping", False),
                "metadata": result.get("metadata", {}),
            }

        except Exception as e:
            log_error(logger, "orchestrator_processing_failed", error_type=type(e).__name__, error=str(e))
            logger.exception("orchestrator_processing_exception")
            return {
                "response": f"An error occurred: {str(e)}",
                "justification": None,
                "sources": [],
                "route_taken": "error",
                "success": False,
                "error": str(e),
            }


def create_orchestrator(
    api_key: str,
    rewriter: Optional[QueryRewriter] = None,
    retriever: Optional[retrivalModel] = None,
    reranker: Optional[ChunkReranker] = None,
    answer_generator: Optional[AnswerGenerator] = None,
) -> QueryOrchestrator:
    return QueryOrchestrator(
        api_key=api_key,
        rewriter=rewriter,
        retriever=retriever,
        reranker=reranker,
        answer_generator=answer_generator,
    )
