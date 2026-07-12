import os
import re
from pathlib import Path

from dotenv import load_dotenv
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from shared.chroma_config import get_shared_collection_name
from shared.logging_utils import get_logger, log_error, log_info


logger = get_logger(__name__)


class retrivalModel:
    def __init__(self):
        env_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(env_path)

        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        api_key = os.getenv("CHROMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError(f"CHROMA_CLOUD_API_KEY is missing. Check {env_path}.")

        self.client = chromadb.CloudClient(
            api_key=api_key,
            tenant='a92961b0-ea65-4a82-a7ad-321a4baaaa60',
            database='Major-Project'
            )
        self._keyword_index_cache: dict[tuple[str, str | None], dict[str, Any]] = {}
        log_info(logger, "retriever_initialized", tenant="a92961b0-ea65-4a82-a7ad-321a4baaaa60", database="Major-Project")

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", (text or "").lower())

    def _build_search_text(self, text: str, metadata: dict | None) -> str:
        metadata = metadata or {}
        parts = [
            str(metadata.get("document_name", "")),
            str(metadata.get("section_heading", "")),
            str(metadata.get("clause_id", "")),
            str(text or ""),
        ]
        return " ".join(part.strip() for part in parts if part and str(part).strip())

    def _resolve_collection_name(self, collection_name: str | None) -> str:
        return collection_name or get_shared_collection_name()

    def _get_collection(self, collection_name: str | None):
        return self.client.get_collection(name=self._resolve_collection_name(collection_name))

    def _get_all_chunks(self, collection_name: str | None = None, document_filter: str = None) -> list[dict]:
        try:
            collection = self._get_collection(collection_name)
            get_params: dict[str, Any] = {
                "include": ["metadatas", "documents"],
            }
            if document_filter:
                get_params["where"] = {"document_name": document_filter}

            results = collection.get(**get_params)
            ids = results.get("ids") or []
            documents = results.get("documents") or []
            metadatas = results.get("metadatas") or []

            chunks: list[dict] = []
            for idx, chunk_id in enumerate(ids):
                chunks.append(
                    {
                        "text": documents[idx],
                        "metadata": metadatas[idx] or {},
                        "chunk_id": chunk_id,
                    }
                )
            return chunks
        except Exception as e:
            log_error(
                logger,
                "keyword_chunk_fetch_failed",
                collection_name=collection_name,
                document_filter=document_filter,
                error_type=type(e).__name__,
                error=str(e),
            )
            return []

    def _get_keyword_index(self, collection_name: str | None = None, document_filter: str = None) -> dict[str, Any] | None:
        resolved_collection_name = self._resolve_collection_name(collection_name)
        cache_key = (resolved_collection_name, document_filter)
        if cache_key in self._keyword_index_cache:
            return self._keyword_index_cache[cache_key]

        chunks = self._get_all_chunks(
            collection_name=resolved_collection_name,
            document_filter=document_filter,
        )
        if not chunks:
            return None

        tokenized_corpus: list[list[str]] = []
        valid_chunks: list[dict] = []
        for chunk in chunks:
            search_text = self._build_search_text(chunk.get("text", ""), chunk.get("metadata"))
            tokens = self._tokenize(search_text)
            if not tokens:
                continue
            tokenized_corpus.append(tokens)
            valid_chunks.append(chunk)

        if not valid_chunks:
            return None

        index = {
            "bm25": BM25Okapi(tokenized_corpus),
            "chunks": valid_chunks,
        }
        self._keyword_index_cache[cache_key] = index
        return index

    def semantic_search(self, rewritten_query: str, collection_name: str | None = None, top_k: int = 15, document_filter: str = None) -> list[dict]:
        try:
            collection = self._get_collection(collection_name)
            query_embedding = self.model.encode([rewritten_query]).tolist()

            query_params = {
                "query_embeddings": query_embedding,
                "n_results": top_k,
                "include": ["metadatas", "documents", "distances"],
            }
            if document_filter:
                query_params["where"] = {"document_name": document_filter}
                log_info(
                    logger,
                    "semantic_search_filter_applied",
                    collection_name=collection_name,
                    document_filter=document_filter,
                )

            results = collection.query(**query_params)

            chunks: list[dict] = []
            if results and results.get("ids") and len(results["ids"]) > 0:
                for idx in range(len(results["ids"][0])):
                    distance = results["distances"][0][idx]
                    chunk = {
                        "text": results["documents"][0][idx],
                        "metadata": results["metadatas"][0][idx],
                        "distance": distance,
                        "semantic_score": 1 / (1 + distance),
                        "chunk_id": results["ids"][0][idx],
                        "matched_by": ["semantic"],
                    }
                    chunks.append(chunk)

                chunks = sorted(chunks, key=lambda x: x["distance"])
                log_info(
                    logger,
                    "semantic_search_completed",
                    collection_name=collection_name,
                    chunk_count=len(chunks),
                    document_filter=document_filter,
                )
            return chunks
        except Exception as e:
            log_error(
                logger,
                "semantic_search_failed",
                collection_name=collection_name,
                document_filter=document_filter,
                error_type=type(e).__name__,
                error=str(e),
            )
            return []

    def keyword_search(self, rewritten_query: str, collection_name: str | None = None, top_k: int = 15, document_filter: str = None) -> list[dict]:
        try:
            index = self._get_keyword_index(collection_name=collection_name, document_filter=document_filter)
            if not index:
                return []

            query_tokens = self._tokenize(rewritten_query)
            if not query_tokens:
                return []

            bm25 = index["bm25"]
            chunks = index["chunks"]
            scores = bm25.get_scores(query_tokens)
            ranked_pairs = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

            keyword_chunks: list[dict] = []
            for idx, score in ranked_pairs[:top_k]:
                if score <= 0:
                    continue
                chunk = dict(chunks[idx])
                chunk["keyword_score"] = float(score)
                chunk["matched_by"] = ["keyword"]
                keyword_chunks.append(chunk)

            log_info(
                logger,
                "keyword_search_completed",
                collection_name=collection_name,
                chunk_count=len(keyword_chunks),
                document_filter=document_filter,
            )
            return keyword_chunks
        except Exception as e:
            log_error(
                logger,
                "keyword_search_failed",
                collection_name=collection_name,
                document_filter=document_filter,
                error_type=type(e).__name__,
                error=str(e),
            )
            return []

    def hybrid_search(self, rewritten_query: str, collection_name: str | None = None, top_k: int = 15, document_filter: str = None) -> list[dict]:
        semantic_k = max(top_k * 2, 15)
        keyword_k = max(top_k * 2, 15)

        semantic_chunks = self.semantic_search(
            rewritten_query,
            collection_name=collection_name,
            top_k=semantic_k,
            document_filter=document_filter,
        )
        keyword_chunks = self.keyword_search(
            rewritten_query,
            collection_name=collection_name,
            top_k=keyword_k,
            document_filter=document_filter,
        )

        if not keyword_chunks:
            return semantic_chunks[:top_k]
        if not semantic_chunks:
            return keyword_chunks[:top_k]

        fused: dict[str, dict] = {}
        rrf_k = 60

        for rank, chunk in enumerate(semantic_chunks, start=1):
            chunk_id = chunk["chunk_id"]
            entry = fused.setdefault(chunk_id, dict(chunk))
            entry["hybrid_score"] = entry.get("hybrid_score", 0.0) + 1 / (rrf_k + rank)
            entry["matched_by"] = sorted(set(entry.get("matched_by", [])) | {"semantic"})

        for rank, chunk in enumerate(keyword_chunks, start=1):
            chunk_id = chunk["chunk_id"]
            entry = fused.setdefault(chunk_id, dict(chunk))
            entry["hybrid_score"] = entry.get("hybrid_score", 0.0) + 1 / (rrf_k + rank)
            entry["keyword_score"] = chunk.get("keyword_score")
            entry["matched_by"] = sorted(set(entry.get("matched_by", [])) | {"keyword"})

        merged_chunks = sorted(
            fused.values(),
            key=lambda item: item.get("hybrid_score", 0.0),
            reverse=True,
        )[:top_k]
        log_info(
            logger,
            "hybrid_search_completed",
            collection_name=collection_name,
            chunk_count=len(merged_chunks),
            document_filter=document_filter,
        )
        return merged_chunks

    def retrive_Chunks(self, rewritten_query: str, collection_name: str | None = None, top_k: int = 15, document_filter: str = None):
        """Retrieve relevant document chunks using hybrid search over semantic and keyword retrieval."""
        try:
            return self.hybrid_search(
                rewritten_query,
                collection_name=collection_name,
                top_k=top_k,
                document_filter=document_filter,
            )
        except Exception as e:
            log_error(
                logger,
                "retrieval_failed",
                collection_name=collection_name,
                document_filter=document_filter,
                error_type=type(e).__name__,
                error=str(e),
            )
            return []

    def get_context_string(self, chunks: list) -> str:
        """
        Convert retrieved chunks into a single context string,
        sorted by relevance.

        Args:
            chunks: List of chunk dictionaries from retrieve_relevant_chunks

        Returns:
            A formatted string containing all chunk contents with metadata
        """
        if not chunks:
            return ""

        context_parts = []
        for chunk in chunks:
            section = f"[Document: {chunk['metadata']['document_name']}]\n"
            section += f"[Section: {chunk['metadata']['section_heading']}]\n"
            section += f"Content: {chunk['text']}\n"
            section += "-" * 80 + "\n"
            context_parts.append(section)

        return "\n".join(context_parts)
