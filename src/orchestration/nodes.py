"""
LangGraph Node Implementations for the RAG Orchestration System.

This module contains the three main processing nodes:
1. Direct LLM Node - Handles queries that don't need document lookup
2. RAG Process Node - Handles queries requiring document retrieval
3. Web Scraping Node - Handles low similarity score scenarios (placeholder)
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from mistralai import Mistral
from queryRewriter.rewriting import QueryRewriter
from retriever.retrival import retrivalModel
from retriever.reranking_mistral import ChunkReranker
from output.answerGeneration_mistral import AnswerGenerator


class DirectLLMNode:
    """
    Node for handling queries that don't require document lookup.
    Provides direct conversational responses using the LLM.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Direct LLM Node.
        
        Args:
            api_key: Mistral API key
        """
        self.client = Mistral(api_key=api_key)
        self.model = "mistral-tiny"
        
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
        """
        Process a query using direct LLM response.
        
        Args:
            query: The user's original query
            
        Returns:
            Dict containing the response and metadata
        """
        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
            
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            if response and response.choices:
                answer = response.choices[0].message.content.strip()
                return {
                    "answer": answer,
                    "justification": None,
                    "sources": [],
                    "route_taken": "direct_llm",
                    "success": True
                }
            else:
                return {
                    "answer": "I apologize, but I couldn't generate a response. Please try again.",
                    "justification": None,
                    "sources": [],
                    "route_taken": "direct_llm",
                    "success": False
                }
                
        except Exception as e:
            print(f"[DirectLLMNode] Error: {str(e)}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "justification": None,
                "sources": [],
                "route_taken": "direct_llm",
                "success": False,
                "error": str(e)
            }


class RAGProcessNode:
    """
    Node for handling queries that require document retrieval and RAG processing.
    Uses the full RAG pipeline: Query Rewriting -> Retrieval -> Reranking -> Answer Generation
    """
    
    # Threshold for triggering web scraping (when distance is too high, meaning low similarity)
    # ChromaDB uses cosine distance: 0 = identical, 2 = opposite
    # Distance > 1.2 indicates poor similarity (less than ~40% similar)
    DISTANCE_THRESHOLD = 1.2
    
    def __init__(
        self,
        api_key: str,
        rewriter: Optional[QueryRewriter] = None,
        retriever: Optional[retrivalModel] = None,
        reranker: Optional[ChunkReranker] = None,
        answer_generator: Optional[AnswerGenerator] = None
    ):
        """
        Initialize the RAG Process Node.
        
        Args:
            api_key: Mistral API key
            rewriter: Optional pre-initialized QueryRewriter
            retriever: Optional pre-initialized retrivalModel
            reranker: Optional pre-initialized ChunkReranker
            answer_generator: Optional pre-initialized AnswerGenerator
        """
        self.api_key = api_key
        self.rewriter = rewriter or QueryRewriter(api_key)
        self.retriever = retriever or retrivalModel()
        self.reranker = reranker or ChunkReranker(api_key)
        self.answer_generator = answer_generator or AnswerGenerator(api_key)
    
    async def process(
        self,
        query: str,
        scope: str = "shared",
        username: Optional[str] = None,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the full RAG pipeline.
        
        Args:
            query: The user's original query
            scope: The search scope ("shared", "personal", "combined")
            username: Username for personal document access
            collection_name: Optional specific collection name
            
        Returns:
            Dict containing the answer, justification, sources, and metadata
        """
        try:
            # Step 1: Query Rewriting
            print(f"[RAGProcessNode] Step 1: Rewriting query...")
            rewritten_query = await self.rewriter.rewrite_query(query)
            if not rewritten_query:
                rewritten_query = query
            print(f"[RAGProcessNode] Rewritten: {rewritten_query}")
            
            # Step 2: Retrieval
            print(f"[RAGProcessNode] Step 2: Retrieving chunks...")
            chunks = self._retrieve_chunks(rewritten_query, scope, username, collection_name)
            
            if not chunks:
                return {
                    "answer": "No relevant policy content found for this question.",
                    "justification": None,
                    "sources": [],
                    "route_taken": "rag",
                    "rewritten_query": rewritten_query,
                    "success": True,
                    "needs_web_scraping": False
                }
            
            # Check distance for web scraping trigger
            # ChromaDB returns distance (lower = better match, higher = worse match)
            top_chunk_distance = chunks[0].get('distance', 2.0) if chunks else 2.0
            needs_web_scraping = top_chunk_distance > self.DISTANCE_THRESHOLD
            
            if needs_web_scraping:
                print(f"[RAGProcessNode] High distance ({top_chunk_distance:.3f} > {self.DISTANCE_THRESHOLD}), flagging for web scraping")
            else:
                print(f"[RAGProcessNode] Good match (distance: {top_chunk_distance:.3f}, threshold: {self.DISTANCE_THRESHOLD})")
            
            # Step 3: Reranking
            print(f"[RAGProcessNode] Step 3: Reranking chunks...")
            reranked_chunks = await self.reranker.rerank_chunks(rewritten_query, chunks, top_k=5)
            
            # Step 4: Answer Generation
            print(f"[RAGProcessNode] Step 4: Generating answer...")
            answer_result = await self.answer_generator.generate_answer(rewritten_query, reranked_chunks)
            
            return {
                "answer": answer_result.get("answer", ""),
                "justification": answer_result.get("justification"),
                "sources": answer_result.get("source_chunks", []),
                "route_taken": "rag",
                "rewritten_query": rewritten_query,
                "success": True,
                "needs_web_scraping": needs_web_scraping,
                "top_chunk_distance": top_chunk_distance  # Raw distance from ChromaDB
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
                "needs_web_scraping": False
            }
    
    def _retrieve_chunks(
        self,
        rewritten_query: str,
        scope: str,
        username: Optional[str],
        collection_name: Optional[str]
    ) -> List[Dict]:
        """
        Retrieve chunks based on scope and collection settings.
        """
        if collection_name:
            return self.retriever.retrive_Chunks(rewritten_query, collection_name=collection_name)
        
        chunks = []
        if scope == "shared":
            chunks = self.retriever.retrive_Chunks(rewritten_query, collection_name="policy_documents")
        elif scope == "personal" and username:
            chunks = self.retriever.retrive_Chunks(
                rewritten_query,
                collection_name=f"user_{username}_documents"
            )
        elif scope == "combined" and username:
            chunks = self.retriever.retrive_Chunks(rewritten_query, collection_name="policy_documents")
            chunks += self.retriever.retrive_Chunks(
                rewritten_query,
                collection_name=f"user_{username}_documents"
            )
        else:
            chunks = self.retriever.retrive_Chunks(rewritten_query, collection_name="policy_documents")
        
        return chunks


class WebScrapingNode:
    """
    Node for handling queries when RAG retrieval returns low similarity scores.
    This is a PLACEHOLDER implementation for future web scraping functionality.
    
    Purpose: When the similarity score after fetching the top chunk is less than
    the threshold (0.5), this node can be triggered to search for information
    on the web.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Web Scraping Node.
        
        Args:
            api_key: API key for any required services
        """
        self.api_key = api_key
        # Placeholder for web scraping setup
        # Future: Add Serper, Tavily, or other web search APIs
    
    async def process(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a query by searching the web for information.
        
        NOTE: This is a placeholder implementation. In a full implementation,
        this would:
        1. Search the web for relevant information
        2. Scrape and extract content
        3. Synthesize an answer from web sources
        
        Args:
            query: The user's query
            context: Optional context from previous processing
            
        Returns:
            Dict containing the answer and metadata
        """
        # Placeholder implementation
        print(f"[WebScrapingNode] Placeholder - Web scraping not yet implemented")
        
        return {
            "answer": (
                "I couldn't find specific information in the policy documents for your query. "
                "Web search functionality is not yet enabled. "
                "Please try rephrasing your question or contact customer support for assistance."
            ),
            "justification": None,
            "sources": [],
            "route_taken": "web_scraping",
            "success": False,
            "message": "Web scraping node is a placeholder - not yet implemented"
        }


# Convenience functions for LangGraph node integration
async def direct_llm_node(state: Dict, api_key: str) -> Dict:
    """
    LangGraph-compatible wrapper for DirectLLMNode.
    """
    node = DirectLLMNode(api_key)
    result = await node.process(state["query"])
    return {**state, "result": result}


async def rag_process_node(
    state: Dict,
    api_key: str,
    rewriter: Optional[QueryRewriter] = None,
    retriever: Optional[retrivalModel] = None,
    reranker: Optional[ChunkReranker] = None,
    answer_generator: Optional[AnswerGenerator] = None
) -> Dict:
    """
    LangGraph-compatible wrapper for RAGProcessNode.
    """
    node = RAGProcessNode(
        api_key,
        rewriter=rewriter,
        retriever=retriever,
        reranker=reranker,
        answer_generator=answer_generator
    )
    result = await node.process(
        state["query"],
        scope=state.get("scope", "shared"),
        username=state.get("username"),
        collection_name=state.get("collection_name")
    )
    return {**state, "result": result}


async def web_scraping_node(state: Dict, api_key: str) -> Dict:
    """
    LangGraph-compatible wrapper for WebScrapingNode.
    """
    node = WebScrapingNode(api_key)
    result = await node.process(state["query"], context=state.get("result"))
    return {**state, "result": result}
