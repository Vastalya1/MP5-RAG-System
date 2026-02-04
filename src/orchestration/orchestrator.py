"""
LangGraph Orchestrator for RAG System

This module implements the main orchestration logic using LangGraph.
It routes queries between Direct LLM and RAG processing paths based on
classification performed BEFORE query rewriting.
"""

from typing import TypedDict, Literal, Optional, Any, Dict, List, Annotated
from typing_extensions import TypedDict
import operator

from langgraph.graph import StateGraph, START, END

from .classifier import QueryClassifier
from .nodes import DirectLLMNode, RAGProcessNode, WebScrapingNode

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from queryRewriter.rewriting import QueryRewriter
from retriever.retrival import retrivalModel
from retriever.reranking_mistral import ChunkReranker
from output.answerGeneration_mistral import AnswerGenerator


class GraphState(TypedDict):
    """
    State schema for the LangGraph orchestration.
    
    Attributes:
        query: The original user query
        rewritten_query: Query after rewriting (for RAG path)
        scope: Search scope (shared, personal, combined)
        username: Username for personal document access
        collection_name: Optional specific collection name
        route: The classified route (rag or direct)
        answer: The generated answer
        justification: Justification for the answer
        sources: Source chunks used for the answer
        success: Whether the processing was successful
        route_taken: Which route was actually taken
        needs_web_scraping: Flag for low similarity scenarios
        error: Error message if any
        metadata: Additional metadata
    """
    query: str
    rewritten_query: Optional[str]
    scope: str
    username: Optional[str]
    collection_name: Optional[str]
    route: Optional[Literal["rag", "direct"]]
    answer: Optional[str]
    justification: Optional[str]
    sources: List[Dict]
    success: bool
    route_taken: Optional[str]
    needs_web_scraping: bool
    error: Optional[str]
    metadata: Dict[str, Any]


class QueryOrchestrator:
    """
    Main orchestrator class that uses LangGraph to route and process queries.
    
    The orchestration flow:
    1. Classify the query (before any rewriting)
    2. Route to either:
       - Direct LLM: For general queries
       - RAG Process: For document-related queries
    3. (Future) Web Scraping: For low similarity RAG results
    """
    
    def __init__(
        self,
        api_key: str,
        rewriter: Optional[QueryRewriter] = None,
        retriever: Optional[retrivalModel] = None,
        reranker: Optional[ChunkReranker] = None,
        answer_generator: Optional[AnswerGenerator] = None
    ):
        """
        Initialize the Query Orchestrator.
        
        Args:
            api_key: Mistral API key
            rewriter: Optional pre-initialized QueryRewriter
            retriever: Optional pre-initialized retrivalModel
            reranker: Optional pre-initialized ChunkReranker
            answer_generator: Optional pre-initialized AnswerGenerator
        """
        self.api_key = api_key
        
        # Initialize components
        self.classifier = QueryClassifier(api_key)
        self.direct_llm_node = DirectLLMNode(api_key)
        self.rag_node = RAGProcessNode(
            api_key,
            rewriter=rewriter,
            retriever=retriever,
            reranker=reranker,
            answer_generator=answer_generator
        )
        self.web_scraping_node = WebScrapingNode(api_key)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph with nodes and edges.
        
        Graph Structure:
            START
              |
              v
        [classifier_node]
              |
              +-- "rag" --> [rag_node] --> END
              |
              +-- "direct" --> [direct_llm_node] --> END
        
        Note: Web scraping node is defined but not connected in edges yet.
        """
        # Create the graph with our state schema
        builder = StateGraph(GraphState)
        
        # Add nodes
        builder.add_node("classifier_node", self._classifier_node)
        builder.add_node("direct_llm_node", self._direct_llm_node)
        builder.add_node("rag_node", self._rag_node)
        builder.add_node("web_scraping_node", self._web_scraping_node)  # Defined but not connected
        
        # Add edges
        # Start with classifier
        builder.add_edge(START, "classifier_node")
        
        # Conditional routing based on classification
        builder.add_conditional_edges(
            "classifier_node",
            self._route_query,
            {
                "direct": "direct_llm_node",
                "rag": "rag_node"
            }
        )
        
        # Both processing nodes go to END
        builder.add_edge("direct_llm_node", END)
        builder.add_edge("rag_node", END)
        
        # Note: web_scraping_node is not connected yet
        # Future: Add conditional edge from rag_node based on similarity threshold
        # builder.add_conditional_edges(
        #     "rag_node",
        #     self._check_similarity_threshold,
        #     {
        #         "proceed": END,
        #         "web_scrape": "web_scraping_node"
        #     }
        # )
        # builder.add_edge("web_scraping_node", END)
        
        # Compile the graph
        compiled_graph = builder.compile()
        
        print("[Orchestrator] LangGraph compiled successfully")
        return compiled_graph
    
    def _classifier_node(self, state: GraphState) -> dict:
        """
        Node that classifies the query to determine routing.
        This happens BEFORE any query rewriting.
        """
        query = state["query"]
        print(f"[Orchestrator] Classifying query: {query[:50]}...")
        
        route = self.classifier.classify(query)
        
        return {
            "route": route,
            "metadata": {**state.get("metadata", {}), "classified_route": route}
        }
    
    async def _direct_llm_node(self, state: GraphState) -> dict:
        """
        Node that processes queries using direct LLM response.
        """
        query = state["query"]
        print(f"[Orchestrator] Processing via Direct LLM...")
        
        result = await self.direct_llm_node.process(query)
        
        return {
            "answer": result.get("answer"),
            "justification": result.get("justification"),
            "sources": result.get("sources", []),
            "success": result.get("success", False),
            "route_taken": "direct_llm",
            "needs_web_scraping": False,
            "error": result.get("error")
        }
    
    async def _rag_node(self, state: GraphState) -> dict:
        """
        Node that processes queries using the full RAG pipeline.
        """
        query = state["query"]
        scope = state.get("scope", "shared")
        username = state.get("username")
        collection_name = state.get("collection_name")
        
        print(f"[Orchestrator] Processing via RAG pipeline...")
        
        result = await self.rag_node.process(
            query=query,
            scope=scope,
            username=username,
            collection_name=collection_name
        )
        
        return {
            "answer": result.get("answer"),
            "justification": result.get("justification"),
            "sources": result.get("sources", []),
            "success": result.get("success", False),
            "route_taken": "rag",
            "rewritten_query": result.get("rewritten_query"),
            "needs_web_scraping": result.get("needs_web_scraping", False),
            "error": result.get("error"),
            "metadata": {
                **state.get("metadata", {}),
                "top_chunk_distance": result.get("top_chunk_distance")
            }
        }
    
    async def _web_scraping_node(self, state: GraphState) -> dict:
        """
        Node that handles web scraping for low similarity scenarios.
        Currently a placeholder.
        """
        query = state["query"]
        print(f"[Orchestrator] Processing via Web Scraping (placeholder)...")
        
        result = await self.web_scraping_node.process(query, context={"previous_result": state})
        
        return {
            "answer": result.get("answer"),
            "justification": result.get("justification"),
            "sources": result.get("sources", []),
            "success": result.get("success", False),
            "route_taken": "web_scraping",
            "error": result.get("error")
        }
    
    def _route_query(self, state: GraphState) -> Literal["direct", "rag"]:
        """
        Routing function for conditional edges.
        Returns the route determined by the classifier.
        """
        route = state.get("route", "rag")
        print(f"[Orchestrator] Routing to: {route}")
        return route
    
    def _check_similarity_threshold(self, state: GraphState) -> Literal["proceed", "web_scrape"]:
        """
        Check if similarity score is below threshold.
        Used for future web scraping routing.
        """
        if state.get("needs_web_scraping", False):
            return "web_scrape"
        return "proceed"
    
    async def process_query(
        self,
        query: str,
        scope: str = "shared",
        username: Optional[str] = None,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the orchestration graph.
        
        Args:
            query: The user's query
            scope: Search scope (shared, personal, combined)
            username: Username for personal document access
            collection_name: Optional specific collection name
            
        Returns:
            Dict containing the answer, justification, sources, and metadata
        """
        # Initialize the state
        initial_state: GraphState = {
            "query": query,
            "rewritten_query": None,
            "scope": scope,
            "username": username,
            "collection_name": collection_name,
            "route": None,
            "answer": None,
            "justification": None,
            "sources": [],
            "success": False,
            "route_taken": None,
            "needs_web_scraping": False,
            "error": None,
            "metadata": {}
        }
        
        try:
            # Invoke the graph
            print(f"[Orchestrator] Starting query processing...")
            result = await self.graph.ainvoke(initial_state)
            
            print(f"[Orchestrator] Processing complete. Route: {result.get('route_taken')}")
            
            return {
                "response": result.get("answer", ""),
                "justification": result.get("justification"),
                "sources": result.get("sources", []),
                "route_taken": result.get("route_taken"),
                "rewritten_query": result.get("rewritten_query"),
                "success": result.get("success", False),
                "needs_web_scraping": result.get("needs_web_scraping", False),
                "metadata": result.get("metadata", {})
            }
            
        except Exception as e:
            print(f"[Orchestrator] Error during processing: {str(e)}")
            return {
                "response": f"An error occurred: {str(e)}",
                "justification": None,
                "sources": [],
                "route_taken": "error",
                "success": False,
                "error": str(e)
            }
    
    def get_graph_visualization(self) -> str:
        """
        Get a text representation of the graph structure.
        
        Returns:
            String representation of the graph
        """
        return """
        LangGraph Orchestration Flow:
        =============================
        
                    START
                      |
                      v
              [classifier_node]
                      |
            +---------+---------+
            |                   |
            v                   v
     (route="direct")    (route="rag")
            |                   |
            v                   v
    [direct_llm_node]    [rag_node]
            |                   |
            +---------+---------+
                      |
                      v
                     END
        
        [web_scraping_node] - Defined but not connected (placeholder)
        
        Future Enhancement:
        - Connect web_scraping_node after rag_node when similarity < threshold
        """


def create_orchestrator(
    api_key: str,
    rewriter: Optional[QueryRewriter] = None,
    retriever: Optional[retrivalModel] = None,
    reranker: Optional[ChunkReranker] = None,
    answer_generator: Optional[AnswerGenerator] = None
) -> QueryOrchestrator:
    """
    Factory function to create a QueryOrchestrator instance.
    
    Args:
        api_key: Mistral API key
        rewriter: Optional pre-initialized QueryRewriter
        retriever: Optional pre-initialized retrivalModel
        reranker: Optional pre-initialized ChunkReranker
        answer_generator: Optional pre-initialized AnswerGenerator
        
    Returns:
        Configured QueryOrchestrator instance
    """
    return QueryOrchestrator(
        api_key=api_key,
        rewriter=rewriter,
        retriever=retriever,
        reranker=reranker,
        answer_generator=answer_generator
    )
