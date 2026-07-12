import os
from threading import Lock
from pathlib import Path
from typing import Any, List, Dict

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from .chunker import chunk_pdfs
from shared.chroma_config import get_shared_collection_name
from shared.logging_utils import get_logger, log_error, log_info


logger = get_logger(__name__)


class DocumentEmbedder:
    def __init__(self, collection_name: str | None = None):
        """Initialize the embedder with SBERT model and ChromaDB."""
        env_path = Path(__file__).resolve().parents[2] / ".env"
        load_dotenv(env_path)

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._embed_lock = Lock()

        api_key = os.getenv("CHROMA_CLOUD_API_KEY")
        if not api_key:
            raise ValueError(f"CHROMA_CLOUD_API_KEY is missing. Check {env_path}.")

        self.client = chromadb.CloudClient(
            api_key=api_key,
            tenant='a92961b0-ea65-4a82-a7ad-321a4baaaa60',
            database='Major-Project'
            )

        resolved_collection_name = collection_name or get_shared_collection_name()
        self.collection = self.client.get_or_create_collection(
            name=resolved_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        log_info(logger, "embedder_initialized", collection_name=resolved_collection_name)

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

    def _prepare_records(self, records: List[Dict]) -> List[Dict]:
        prepared: List[Dict] = []
        for record in records:
            text = str(record.get("text", "")).strip()
            chunk_id = str(record.get("chunk_id", "")).strip()
            if not text or not chunk_id:
                continue

            metadata = record.get("metadata") or {}
            prepared.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": self._sanitize_metadata(metadata),
                }
            )
        return prepared

    def embed_records(self, records: List[Dict], batch_size: int = 300) -> None:
        """Embed arbitrary text records and store them in ChromaDB in batches."""
        prepared_records = self._prepare_records(records)
        total = len(prepared_records)
        if total == 0:
            log_info(logger, "embed_records_skipped_no_valid_chunks")
            return

        for i in range(0, total, batch_size):
            batch = prepared_records[i:i + batch_size]
            texts = [record["text"] for record in batch]
            ids = [record["chunk_id"] for record in batch]
            metadatas = [record["metadata"] for record in batch]

            with self._embed_lock:
                embeddings = self.model.encode(texts)

                # upsert avoids duplicate-id failures on re-ingestion of the same file.
                self.collection.upsert(
                    documents=texts,
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas,
                )
            log_info(
                logger,
                "embed_batch_completed",
                batch_number=i // batch_size + 1,
                batch_size=len(batch),
            )
        log_info(logger, "embed_records_completed", total_chunks=total)

    def embed_documents(self, chunks: List[Dict], batch_size: int = 300) -> None:
        """Embed document chunks and store them in ChromaDB in batches to avoid quota errors."""
        self.embed_records(chunks, batch_size=batch_size)

    def process_pdf_folder(self, input_folder: str) -> None:
        """Process PDFs in a folder, embed them, and store in ChromaDB."""
        chunks = chunk_pdfs(input_folder)
        self.embed_documents(chunks)
        log_info(logger, "pdf_folder_processed", input_folder=input_folder, chunk_count=len(chunks))
