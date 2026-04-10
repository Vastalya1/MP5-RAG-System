import os
import re
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi


class retrivalModel:
    def __init__(self):
        self.client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_CLOUD_API_KEY"),
            tenant="a92961b0-ea65-4a82-a7ad-321a4baaaa60",
            database="Major-Project",
        )
        self._keyword_index_cache: dict[tuple[str, str | None], dict[str, Any]] = {}

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

    def _get_collection(self, collection_name: str):
        return self.client.get_collection(name=collection_name)

    def _get_all_chunks(
        self,
        collection_name: str = "dataset",
        document_filter: str = None,
    ) -> list[dict]:
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
            print(f"Error fetching all chunks for lexical retrieval: {str(e)}")
            return []

    def _get_keyword_index(
        self,
        collection_name: str = "dataset",
        document_filter: str = None,
    ) -> dict[str, Any] | None:
        cache_key = (collection_name, document_filter)
        if cache_key in self._keyword_index_cache:
            return self._keyword_index_cache[cache_key]

        chunks = self._get_all_chunks(
            collection_name=collection_name,
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

    def lexical_search(
        self,
        rewritten_query: str,
        collection_name: str = "dataset",
        top_k: int = 15,
        document_filter: str = None,
    ) -> list[dict]:
        try:
            index = self._get_keyword_index(
                collection_name=collection_name,
                document_filter=document_filter,
            )
            if not index:
                return []

            query_tokens = self._tokenize(rewritten_query)
            if not query_tokens:
                return []

            bm25 = index["bm25"]
            chunks = index["chunks"]
            scores = bm25.get_scores(query_tokens)
            ranked_pairs = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)

            lexical_chunks: list[dict] = []
            for idx, score in ranked_pairs[:top_k]:
                if score <= 0:
                    continue
                chunk = dict(chunks[idx])
                chunk["keyword_score"] = float(score)
                chunk["lexical_score"] = float(score)
                chunk["matched_by"] = ["keyword"]
                lexical_chunks.append(chunk)

            print(f" Retrieved {len(lexical_chunks)} lexical chunks")
            return lexical_chunks
        except Exception as e:
            print(f"Error in lexical retrieval: {str(e)}")
            return []

    def keyword_search(
        self,
        rewritten_query: str,
        collection_name: str = "dataset",
        top_k: int = 15,
        document_filter: str = None,
    ) -> list[dict]:
        return self.lexical_search(
            rewritten_query,
            collection_name=collection_name,
            top_k=top_k,
            document_filter=document_filter,
        )

    def retrive_Chunks(
        self,
        rewritten_query: str,
        collection_name: str = "dataset",
        top_k: int = 15,
        document_filter: str = None,
    ):
        """Retrieve relevant document chunks using BM25 lexical search."""
        try:
            return self.lexical_search(
                rewritten_query,
                collection_name=collection_name,
                top_k=top_k,
                document_filter=document_filter,
            )
        except Exception as e:
            print(f"Error in retrieval: {str(e)}")
            return []

    def get_context_string(self, chunks: list) -> str:
        """
        Convert retrieved chunks into a single context string,
        sorted by BM25 relevance.

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
