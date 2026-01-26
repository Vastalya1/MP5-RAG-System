import os
from pathlib import Path
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

from .chunker import chunk_pdfs


class DocumentEmbedder:
    def __init__(self, collection_name: str = "policy_documents"):
        """Initialize the embedder with SBERT model and ChromaDB."""
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        base_dir = Path(__file__).resolve().parents[2]
        persist_dir = Path(os.getenv("CHROMA_PERSIST_DIR", str(base_dir / "chromadb")))
        persist_dir.mkdir(parents=True, exist_ok=True)

        # New Chroma client API (PersistentClient).
        self.client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_CLOUD_API_KEY"),
            tenant='a92961b0-ea65-4a82-a7ad-321a4baaaa60',
            database='Major-Project'
            )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )


    def embed_documents(self, chunks: List[Dict], batch_size: int = 300) -> None:
        """Embed document chunks and store them in ChromaDB in batches to avoid quota errors."""
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch = chunks[i:i+batch_size]
            texts = [chunk["text"] for chunk in batch]
            ids = [chunk["chunk_id"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]
            embeddings = self.model.encode(texts)
            self.collection.add(
                documents=texts,
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
            )
            print(f"Embedded and stored batch {i//batch_size + 1} ({len(batch)} chunks) in ChromaDB")
        print(f"Successfully embedded and stored {total} chunks in ChromaDB (in batches)")

    def process_pdf_folder(self, input_folder: str) -> None:
        """Process PDFs in a folder, embed them, and store in ChromaDB."""
        chunks = chunk_pdfs(input_folder)
        self.embed_documents(chunks)
        print("Database persisted to disk")
