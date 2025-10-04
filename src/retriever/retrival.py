from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer


class retrivalModel:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        self.client = chromadb.PersistentClient(
                path=r"D:\_official_\_MIT ADT_\_SEMESTER 7_\MP5\MP5-RAG-System\chromadb",
        )



    def retrive_Chunks(self, rewritten_query: str, collection_name: str="policy_documents", top_k: int =15):
        """Retrieve relevant document chunks from ChromaDB based on the rewritten query.

        Args:
            rewritten_query (str): The query rewritten with medical insurance terminology.
            collection_name (str): The name of the ChromaDB collection to query.
            top_k (int): The number of top relevant chunks to retrieve.

        Returns:
            List[Dict]: List of dictionaries containing chunks and their metadata, sorted by relevance
        """
        try:
            # Get the collection
            collection = self.client.get_collection(name=collection_name)
            
            # Generate embedding for the query
            query_embedding = self.model.encode([rewritten_query]).tolist()

            # Query using HNSW index (ChromaDB uses HNSW by default)
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )

            # Format results into a more usable structure
            chunks = []
            if results and results['ids'] and len(results['ids']) > 0:
                for idx in range(len(results['ids'][0])):
                    chunk = {
                        'text': results['documents'][0][idx],
                        'metadata': results['metadatas'][0][idx],
                        'distance': results['distances'][0][idx],
                        'chunk_id': results['ids'][0][idx]
                    }
                    chunks.append(chunk)
                
                # Sort chunks by distance (most relevant first)
                chunks = sorted(chunks, key=lambda x: x['distance'])
                print(f"✨ Retrieved {len(chunks)} relevant chunks")
            
            return chunks

        except Exception as e:
            print(f"Error in retrieval: {str(e)}")
            return []
            
    def get_context_string(self, chunks: list) -> str:
        """
        Convert retrieved chunks into a single context string,
        sorted by relevance (distance)

        Args:
            chunks: List of chunk dictionaries from retrieve_relevant_chunks

        Returns:
            A formatted string containing all chunk contents with metadata
        """
        if not chunks:
            return ""
        
        # Build context string
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            section = f"[Document: {chunk['metadata']['document_name']}]\n"
            section += f"[Section: {chunk['metadata']['section_heading']}]\n"
            section += f"Content: {chunk['text']}\n"
            section += "-" * 80 + "\n"
            context_parts.append(section)
        
        return "\n".join(context_parts)
        