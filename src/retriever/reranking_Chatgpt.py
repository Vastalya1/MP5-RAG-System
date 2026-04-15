from typing import Dict, List

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class ChunkReranker:
    def __init__(self, api_key: str):
        """Initialize the reranker with SBERT model and OpenAI."""
        # Initialize SBERT model - same as used in retrieval and embedding
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Initialize OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = "gpt-4o-mini"

        # Prompt template for LLM reranking
        self.RERANK_PROMPT = """You are a medical insurance expert tasked with ranking document chunks by relevance to a query.

Query: {query}

Document chunks to rank:
{chunks}

Instructions:
1. Analyze the relevance of each chunk to the query.
2. Select the top 5 most relevant chunks.
3. IMPORTANT: Respond ONLY with the chunk numbers in a comma-separated format.
4. Do not add any explanations, just the numbers.

Example correct responses:
0,1,2,3,4
4,2,0,1,3
1,4,3,2,0

Your response must match this format exactly - just numbers and commas, nothing else.
Response:"""

    def metadata_enhanced_reranking(
        self,
        query: str,
        chunks: List[Dict],
        chunk_weight: float = 0.7,
        heading_weight: float = 0.3,
    ) -> List[Dict]:
        """
        Rerank chunks using combined similarity of chunk content and section headings

        Args:
            query: The search query
            chunks: List of chunks from retrieval
            chunk_weight: Weight for chunk content similarity (default: 0.7)
            heading_weight: Weight for section heading similarity (default: 0.3)

        Returns:
            Reranked list of chunks with updated scores
        """
        try:
            if not chunks:
                return []

            query_embedding = self.model.encode(query)

            for chunk in chunks:
                content_embedding = self.model.encode(chunk["text"])
                content_similarity = np.dot(query_embedding, content_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding)
                )

                heading_embedding = self.model.encode(chunk["metadata"]["section_heading"])
                heading_similarity = np.dot(query_embedding, heading_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(heading_embedding)
                )

                chunk["combined_score"] = (
                    chunk_weight * content_similarity + heading_weight * heading_similarity
                )

            reranked_chunks = sorted(chunks, key=lambda x: x["combined_score"], reverse=True)
            print("Completed metadata-enhanced reranking")
            return reranked_chunks

        except Exception as e:
            print(f"Error in metadata reranking: {str(e)}")
            return chunks

    async def llm_reranking(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Use OpenAI to rerank the chunks based on relevance to query

        Args:
            query: The search query
            chunks: List of chunks to rerank
            top_k: Number of chunks to return (default: 5)

        Returns:
            Top k most relevant chunks according to LLM
        """
        try:
            if not chunks:
                return []

            chunks_text = ""
            for i, chunk in enumerate(chunks):
                chunks_text += f"\nChunk {i}:\n"
                chunks_text += f"Section: {chunk['metadata']['section_heading']}\n"
                chunks_text += f"Content: {chunk['text']}\n"
                chunks_text += "-" * 80 + "\n"

            formatted_prompt = self.RERANK_PROMPT.format(query=query, chunks=chunks_text)

            messages = [
                {
                    "role": "system",
                    "content": "You are a medical insurance expert helping to rank document chunks by relevance.",
                },
                {"role": "user", "content": formatted_prompt},
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                top_p=0.95,
                max_tokens=50,
            )

            if response and response.choices:
                try:
                    response_text = (response.choices[0].message.content or "").strip()
                    response_text = response_text.split("\n")[0]
                    cleaned_text = "".join(
                        char for char in response_text if char.isdigit() or char == ","
                    )
                    indices = [int(idx.strip()) for idx in cleaned_text.split(",") if idx.strip()][:top_k]
                    valid_indices = [idx for idx in indices if idx < len(chunks)]
                    if not valid_indices:
                        print("No valid indices found in response, falling back to default ranking")
                        return chunks[:top_k]

                    reranked_chunks = [chunks[idx] for idx in valid_indices]
                    print(f" Completed LLM reranking, selected {len(reranked_chunks)} chunks")
                    return reranked_chunks
                except Exception as e:
                    print(f"Error parsing LLM response: {str(e)}")
                    return chunks[:top_k]

            print("Error: Empty response from LLM")
            return chunks[:top_k]

        except Exception as e:
            print(f"Error in LLM reranking: {str(e)}")
            return chunks[:top_k]

    def rerank_chunks_sync(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Synchronous reranking pipeline for thread-based execution paths.
        """
        try:
            metadata_reranked = self.metadata_enhanced_reranking(query, chunks)
            return (
                metadata_reranked
                if not metadata_reranked
                else self._llm_rerank_sync(query, metadata_reranked, top_k)
            )
        except Exception as e:
            print(f"Error in synchronous reranking pipeline: {str(e)}")
            return chunks[:top_k]

    def _llm_rerank_sync(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Synchronous wrapper around the LLM reranking step.
        """
        try:
            if not chunks:
                return []

            chunks_text = ""
            for i, chunk in enumerate(chunks):
                chunks_text += f"\nChunk {i}:\n"
                chunks_text += f"Section: {chunk['metadata']['section_heading']}\n"
                chunks_text += f"Content: {chunk['text']}\n"
                chunks_text += "-" * 80 + "\n"

            formatted_prompt = self.RERANK_PROMPT.format(query=query, chunks=chunks_text)

            messages = [
                {
                    "role": "system",
                    "content": "You are a medical insurance expert helping to rank document chunks by relevance.",
                },
                {"role": "user", "content": formatted_prompt},
            ]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                top_p=0.95,
                max_tokens=50,
            )

            if response and response.choices:
                response_text = ((response.choices[0].message.content or "").strip()).split("\n")[0]
                cleaned_text = "".join(char for char in response_text if char.isdigit() or char == ",")
                indices = [int(idx.strip()) for idx in cleaned_text.split(",") if idx.strip()][:top_k]
                valid_indices = [idx for idx in indices if idx < len(chunks)]
                if not valid_indices:
                    print("No valid indices found in response, falling back to default ranking")
                    return chunks[:top_k]
                reranked_chunks = [chunks[idx] for idx in valid_indices]
                print(f" Completed LLM reranking, selected {len(reranked_chunks)} chunks")
                return reranked_chunks

            print("Error: Empty response from LLM")
            return chunks[:top_k]
        except Exception as e:
            print(f"Error in synchronous LLM reranking: {str(e)}")
            return chunks[:top_k]

    async def rerank_chunks(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Complete reranking pipeline: metadata-enhanced followed by LLM reranking

        Args:
            query: The search query
            chunks: Initial chunks from retrieval
            top_k: Final number of chunks to return

        Returns:
            Final reranked list of most relevant chunks
        """
        return self.rerank_chunks_sync(query, chunks, top_k)
