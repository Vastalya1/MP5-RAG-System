from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from mistralai import Mistral

class ChunkReranker:
    def __init__(self, api_key: str):
        """Initialize the reranker with SBERT model and Mistral"""
        # Initialize SBERT model - same as used in retrieval and embedding
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize Mistral
        self.client = Mistral(api_key=api_key)
        self.model_name = "mistral-small"  # Using the same model as query rewriter
        
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

    def metadata_enhanced_reranking(self, 
                                  query: str, 
                                  chunks: List[Dict], 
                                  chunk_weight: float = 0.7, 
                                  heading_weight: float = 0.3) -> List[Dict]:
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

            # Encode query once
            query_embedding = self.model.encode(query)
            
            # Process each chunk
            for chunk in chunks:
                # Get content embedding
                content_embedding = self.model.encode(chunk['text'])
                content_similarity = np.dot(query_embedding, content_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding)
                )
                
                # Get section heading embedding
                heading_embedding = self.model.encode(chunk['metadata']['section_heading'])
                heading_similarity = np.dot(query_embedding, heading_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(heading_embedding)
                )
                
                # Calculate combined score
                chunk['combined_score'] = (
                    chunk_weight * content_similarity + 
                    heading_weight * heading_similarity
                )
            
            # Sort by combined score
            reranked_chunks = sorted(chunks, key=lambda x: x['combined_score'], reverse=True)
            print(f"✨ Completed metadata-enhanced reranking")
            return reranked_chunks
            
        except Exception as e:
            print(f"Error in metadata reranking: {str(e)}")
            return chunks

    async def llm_reranking(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Use Gemini to rerank the chunks based on relevance to query
        
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

            # Prepare chunks for LLM prompt
            chunks_text = ""
            for i, chunk in enumerate(chunks):
                chunks_text += f"\nChunk {i}:\n"
                chunks_text += f"Section: {chunk['metadata']['section_heading']}\n"
                chunks_text += f"Content: {chunk['text']}\n"
                chunks_text += "-" * 80 + "\n"

            # Format prompt
            formatted_prompt = self.RERANK_PROMPT.format(
                query=query,
                chunks=chunks_text
            )

            # Create chat messages
            messages = [
                {"role": "system", "content": "You are a medical insurance expert helping to rank document chunks by relevance."},
                {"role": "user", "content": formatted_prompt}
            ]

            # Get response from Mistral API
            response = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  # Low temperature for consistent ranking
                top_p=0.95,
                max_tokens=50
            )
            
            if response and response.choices:
                # Parse indices from response
                try:
                    response_text = response.choices[0].message.content.strip()
                    
                    # Clean up the response text
                    # Remove any text after newlines and remove any non-numeric/comma characters
                    response_text = response_text.split('\n')[0]
                    cleaned_text = ''.join(char for char in response_text if char.isdigit() or char == ',')
                    
                    # Split and convert to integers, filter out any empty strings
                    indices = [int(idx.strip()) for idx in cleaned_text.split(',') if idx.strip()][:top_k]
                    
                    # Validate indices
                    valid_indices = [idx for idx in indices if idx < len(chunks)]
                    if not valid_indices:
                        print("No valid indices found in response, falling back to default ranking")
                        return chunks[:top_k]
                        
                    indices = valid_indices
                    # Get chunks in the order specified by LLM
                    reranked_chunks = [chunks[idx] for idx in indices if idx < len(chunks)]
                    print(f"✨ Completed LLM reranking, selected {len(reranked_chunks)} chunks")
                    return reranked_chunks
                except Exception as e:
                    print(f"Error parsing LLM response: {str(e)}")
                    return chunks[:top_k]
            else:
                print("Error: Empty response from LLM")
                return chunks[:top_k]

        except Exception as e:
            print(f"Error in LLM reranking: {str(e)}")
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
        try:
            # Step 1: Metadata-enhanced reranking
            metadata_reranked = self.metadata_enhanced_reranking(query, chunks)
            
            # Step 2: LLM reranking of top chunks
            final_chunks = await self.llm_reranking(query, metadata_reranked, top_k)
            
            return final_chunks
            
        except Exception as e:
            print(f"Error in reranking pipeline: {str(e)}")
            return chunks[:top_k]


# Example usage
if __name__ == "__main__":
    import asyncio
    from retrival import retrivalModel
    
    async def main():
        # Initialize components
        retriever = retrivalModel()
        reranker = ChunkReranker(api_key="your-api-key-here")
        
        # Example query
        query = "What are the waiting periods for pre-existing conditions?"
        
        # Get initial chunks
        initial_chunks = retriever.retrive_Chunks(query)
        
        # Rerank chunks
        final_chunks = await reranker.rerank_chunks(query, initial_chunks)
        
        # Print results
        print("\nFinal Reranked Chunks:")
        for i, chunk in enumerate(final_chunks, 1):
            print(f"\nChunk {i}:")
            print(f"Section: {chunk['metadata']['section_heading']}")
            print(f"Score: {chunk.get('combined_score', 'N/A')}")
            print("-" * 80)
            print(chunk['text'])
            print("-" * 80)
    
    asyncio.run(main())
