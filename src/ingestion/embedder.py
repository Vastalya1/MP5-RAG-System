import os
from pathlib import Path
from typing import Any, List, Dict

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from .chunker import chunk_pdfs


class DocumentEmbedder:
    def __init__(self, collection_name: str = "temp_dataset"):
        """Initialize the embedder with SBERT model and ChromaDB."""
        env_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(env_path)

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        api_key = os.getenv("CHROMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError(f"CHROMA_CLOUD_API_KEY is missing. Check {env_path}.")

        self.client = chromadb.CloudClient(
            api_key=api_key,
            tenant='a92961b0-ea65-4a82-a7ad-321a4baaaa60',
            database='Major-Project'
            )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chroma metadata values should be primitive and non-null.
        The chunker may emit clause_id=None for unnumbered headings.
        """
        cleaned: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                cleaned[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)

        if not cleaned.get("document_name"):
            cleaned["document_name"] = "unknown_document"
        if not cleaned.get("section_heading"):
            cleaned["section_heading"] = "General"
        if "clause_id" not in cleaned:
            cleaned["clause_id"] = ""
        return cleaned

    def _prepare_chunks(self, chunks: List[Dict]) -> List[Dict]:
        prepared: List[Dict] = []
        for chunk in chunks:
            text = str(chunk.get("text", "")).strip()
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if not text or not chunk_id:
                continue

            metadata = chunk.get("metadata") or {}
            prepared.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": self._sanitize_metadata(metadata),
                }
            )
        return prepared

    def embed_documents(self, chunks: List[Dict], batch_size: int = 300) -> None:
        """Embed document chunks and store them in ChromaDB in batches to avoid quota errors."""
        prepared_chunks = self._prepare_chunks(chunks)
        total = len(prepared_chunks)
        if total == 0:
            print("No valid chunks to embed.")
            return

        for i in range(0, total, batch_size):
            batch = prepared_chunks[i:i + batch_size]
            texts = [chunk["text"] for chunk in batch]
            ids = [chunk["chunk_id"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]
            embeddings = self.model.encode(texts)

            # upsert avoids duplicate-id failures on re-ingestion of the same file.
            self.collection.upsert(
                documents=texts,
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
            )
            print(f"Embedded and stored batch {i // batch_size + 1} ({len(batch)} chunks) in ChromaDB")
        print(f"Successfully embedded and stored {total} chunks in ChromaDB (in batches)")

    def process_pdf_folder(self, input_folder: str) -> None:
        """Process PDFs in a folder, embed them, and store in ChromaDB."""
        chunks = chunk_pdfs(input_folder)
        self.embed_documents(chunks)
        print("Database persisted to disk")
