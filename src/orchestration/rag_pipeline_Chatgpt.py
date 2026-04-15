"""
Reusable RAG processing utilities shared by orchestration nodes.
"""

from typing import Any, Dict, List, Optional

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from output.answerGeneration_Chatgpt import AnswerGenerator
from queryRewriter.rewriting_Chatgpt import QueryRewriter
from retriever.reranking_Chatgpt import ChunkReranker
from retriever.retrival import retrivalModel
from shared.chroma_config import get_personal_collection_name, get_shared_collection_name


def build_retrieval_debug(chunks: List[Dict]) -> Dict[str, Any]:
    source_counts = {"semantic": 0, "keyword": 0, "both": 0}
    top_chunks: List[Dict[str, Any]] = []

    for chunk in (chunks or [])[:5]:
        matched_by = chunk.get("matched_by", []) or []
        matched_set = set(matched_by)
        if matched_set == {"semantic"}:
            source_counts["semantic"] += 1
        elif matched_set == {"keyword"}:
            source_counts["keyword"] += 1
        elif matched_set:
            source_counts["both"] += 1

        metadata = chunk.get("metadata", {}) or {}
        top_chunks.append(
            {
                "document": metadata.get("document_name", "unknown_document"),
                "section": metadata.get("section_heading", "General"),
                "matched_by": sorted(matched_set),
                "hybrid_score": chunk.get("hybrid_score"),
                "semantic_score": chunk.get("semantic_score"),
                "keyword_score": chunk.get("keyword_score"),
            }
        )

    return {
        "top_chunk_count": len((chunks or [])[:5]),
        "source_counts": source_counts,
        "top_chunks": top_chunks,
    }


def merge_sources(source_groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for group in source_groups:
        for source in group or []:
            key = (
                str(source.get("document", "")),
                str(source.get("section", "")),
                str(source.get("text", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(source)

    return merged


def aggregate_retrieval_debug(subquery_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    source_counts = {"semantic": 0, "keyword": 0, "both": 0}
    top_chunks: List[Dict[str, Any]] = []
    sub_queries: List[Dict[str, Any]] = []

    for result in subquery_results or []:
        debug = result.get("retrieval_debug") or {}
        counts = debug.get("source_counts") or {}
        source_counts["semantic"] += counts.get("semantic", 0)
        source_counts["keyword"] += counts.get("keyword", 0)
        source_counts["both"] += counts.get("both", 0)

        for chunk in debug.get("top_chunks", []) or []:
            top_chunks.append({**chunk, "sub_query": result.get("sub_query")})

        sub_queries.append(
            {
                "sub_query": result.get("sub_query"),
                "rewritten_query": result.get("rewritten_query"),
                "success": result.get("success", False),
                "needs_web_scraping": result.get("needs_web_scraping", False),
                "top_chunk_count": debug.get("top_chunk_count", 0),
            }
        )

    return {
        "decomposition_used": len(subquery_results or []) > 1,
        "subquery_count": len(subquery_results or []),
        "source_counts": source_counts,
        "top_chunks": top_chunks[:5],
        "sub_queries": sub_queries,
    }


class RAGSubQueryProcessor:
    """
    Reusable RAG pipeline for processing one query or decomposed sub-query.
    """

    DISTANCE_THRESHOLD = 1.3

    def __init__(
        self,
        api_key: str,
        rewriter: Optional[QueryRewriter] = None,
        retriever: Optional[retrivalModel] = None,
        reranker: Optional[ChunkReranker] = None,
        answer_generator: Optional[AnswerGenerator] = None,
    ):
        self.api_key = api_key
        self.rewriter = rewriter or QueryRewriter(api_key)
        self.retriever = retriever or retrivalModel()
        self.reranker = reranker or ChunkReranker(api_key)
        self.answer_generator = answer_generator or AnswerGenerator(api_key)

    def process_sync(
        self,
        query: str,
        scope: str = "shared",
        username: Optional[str] = None,
        collection_name: Optional[str] = None,
        document_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            print(f"[RAGSubQueryProcessor] Rewriting query: {query[:80]}")
            rewritten_query = self.rewriter.rewrite_query_sync(query) or query

            print(f"[RAGSubQueryProcessor] Retrieving chunks for: {rewritten_query[:80]}")
            chunks = self.retrieve_chunks(
                rewritten_query=rewritten_query,
                scope=scope,
                username=username,
                collection_name=collection_name,
                document_filter=document_filter,
            )

            if not chunks:
                return {
                    "answer": "No relevant policy content found for this question.",
                    "justification": None,
                    "sources": [],
                    "rewritten_query": rewritten_query,
                    "success": True,
                    "needs_web_scraping": False,
                    "retrieval_debug": build_retrieval_debug([]),
                    "top_chunk_distance": 2.0,
                }

            semantic_distances = [
                chunk.get("distance") for chunk in chunks if chunk.get("distance") is not None
            ]
            top_chunk_distance = min(semantic_distances) if semantic_distances else 2.0
            needs_web_scraping = top_chunk_distance > self.DISTANCE_THRESHOLD

            if needs_web_scraping:
                print(
                    f"[RAGSubQueryProcessor] High distance ({top_chunk_distance:.3f} > "
                    f"{self.DISTANCE_THRESHOLD}), marking low confidence"
                )
            else:
                print(f"[RAGSubQueryProcessor] Good semantic match ({top_chunk_distance:.3f})")

            print(f"[RAGSubQueryProcessor] Reranking chunks for: {rewritten_query[:80]}")
            reranked_chunks = self.reranker.rerank_chunks_sync(rewritten_query, chunks, top_k=5)

            print(f"[RAGSubQueryProcessor] Generating answer for: {rewritten_query[:80]}")
            answer_result = self.answer_generator.generate_answer_sync(rewritten_query, reranked_chunks)

            return {
                "answer": answer_result.get("answer", ""),
                "justification": answer_result.get("justification"),
                "sources": answer_result.get("source_chunks", []),
                "rewritten_query": rewritten_query,
                "success": True,
                "needs_web_scraping": needs_web_scraping,
                "top_chunk_distance": top_chunk_distance,
                "retrieval_debug": build_retrieval_debug(reranked_chunks),
            }

        except Exception as e:
            print(f"[RAGSubQueryProcessor] Error: {str(e)}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "justification": None,
                "sources": [],
                "rewritten_query": query,
                "success": False,
                "error": str(e),
                "needs_web_scraping": False,
                "retrieval_debug": build_retrieval_debug([]),
                "top_chunk_distance": 2.0,
            }

    async def process(
        self,
        query: str,
        scope: str = "shared",
        username: Optional[str] = None,
        collection_name: Optional[str] = None,
        document_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.process_sync(
            query=query,
            scope=scope,
            username=username,
            collection_name=collection_name,
            document_filter=document_filter,
        )

    def retrieve_chunks(
        self,
        rewritten_query: str,
        scope: str,
        username: Optional[str],
        collection_name: Optional[str],
        document_filter: Optional[str],
    ) -> List[Dict]:
        if collection_name:
            return self.retriever.retrive_Chunks(
                rewritten_query,
                collection_name=collection_name,
                document_filter=document_filter,
            )

        if scope == "shared":
            return self.retriever.retrive_Chunks(
                rewritten_query,
                collection_name=get_shared_collection_name(),
                document_filter=document_filter,
            )

        if scope == "personal" and username:
            return self.retriever.retrive_Chunks(
                rewritten_query,
                collection_name=get_personal_collection_name(username),
                document_filter=document_filter,
            )

        if scope == "combined" and username:
            shared_chunks = self.retriever.retrive_Chunks(
                rewritten_query,
                collection_name=get_shared_collection_name(),
                document_filter=document_filter,
            )
            personal_chunks = self.retriever.retrive_Chunks(
                rewritten_query,
                collection_name=get_personal_collection_name(username),
                document_filter=document_filter,
            )
            return shared_chunks + personal_chunks

        return self.retriever.retrive_Chunks(
            rewritten_query,
            collection_name=get_shared_collection_name(),
            document_filter=document_filter,
        )
