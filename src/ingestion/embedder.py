import os
from typing import List, Dict
import os
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from .chunker import chunk_pdfs

class DocumentEmbedder:
    def __init__(self, collection_name: str = "policy_documents"):
        """Initialize the embedder with SBERT model and ChromaDB"""
        # Initialize the SBERT model
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
                path=r"D:\_official_\_MIT ADT_\_SEMESTER 7_\MP5\MP5-RAG-System\chromadb",
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def embed_documents(self, chunks: List[Dict]) -> None:
        """
        Embed document chunks and store them in ChromaDB
        Args:
            chunks: List of dictionaries containing document chunks and metadata
        """
        # Prepare data for ChromaDB
        texts = [chunk["text"] for chunk in chunks]
        ids = [chunk["chunk_id"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Generate embeddings using SBERT
        embeddings = self.model.encode(texts)
        
        # Add to ChromaDB
        self.collection.add(
            documents=texts,
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        print(f"✨ Successfully embedded and stored {len(chunks)} chunks in ChromaDB")

    def process_pdf_folder(self, input_folder: str) -> None:
        """
        Process all PDFs in a folder, create chunks, embed them, and store in ChromaDB
        Args:
            input_folder: Path to folder containing PDF files
        """
        # Get chunks from PDFs
        chunks = chunk_pdfs(input_folder, None)
        
        # Embed and store chunks
        self.embed_documents(chunks)
        
        # Persist the database
        self.client.persist()
        print("💾 Database persisted to disk")


# if __name__ == "__main__":
#     # Example usage
#     input_folder = "D:\_official_\_MIT ADT_\_SEMESTER 7_\MP5\MP5-RAG-System\dataset"  # folder containing your PDFs
#     embedder = DocumentEmbedder()
#     embedder.process_pdf_folder(input_folder)
