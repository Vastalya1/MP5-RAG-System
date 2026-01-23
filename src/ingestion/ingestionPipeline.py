from typing import Optional, List
from pathlib import Path
from .chunker import chunk_pdfs, chunk_pdf_files
from .embedder import DocumentEmbedder


class IngestionPipeline:
    def __init__(
        self,
        dataset_dir: str,
        collection_name: str = "policy_documents",
        chunk_output_dir: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
    ):
        """
        Initialize the ingestion pipeline.

        Args:
            dataset_dir: Directory containing PDF files
            collection_name: Name for the ChromaDB collection
            chunk_output_dir: Optional directory to save JSON chunks
            file_paths: Optional list of PDF file paths to ingest
        """
        self.dataset_dir = Path(dataset_dir)
        self.chunk_output_dir = Path(chunk_output_dir) if chunk_output_dir else None
        self.collection_name = collection_name
        self.file_paths = file_paths

        self.embedder = DocumentEmbedder(collection_name=collection_name)

    def run(self) -> None:
        """
        Run the complete ingestion pipeline:
        1. Chunk PDF documents
        2. Create embeddings
        3. Store in ChromaDB
        """
        if self.file_paths:
            print(f"Processing PDFs: {len(self.file_paths)} file(s)")
        else:
            print(f"Processing PDFs from: {self.dataset_dir}")

        chunk_output_file = None
        if self.chunk_output_dir:
            self.chunk_output_dir.mkdir(parents=True, exist_ok=True)
            chunk_output_file = str(self.chunk_output_dir / "chunks.json")

        try:
            print("Step 1: Chunking PDFs...")
            if self.file_paths:
                chunks = chunk_pdf_files(
                    file_paths=self.file_paths,
                    output_file=chunk_output_file,
                )
            else:
                chunks = chunk_pdfs(
                    input_folder=str(self.dataset_dir),
                    output_file=chunk_output_file,
                )
            print(f"Chunking complete - Created {len(chunks)} chunks")

            print("Step 2: Creating embeddings and storing in ChromaDB...")
            self.embedder.embed_documents(chunks)

            print("Ingestion pipeline completed successfully.")
        except Exception as e:
            print(f"Error during ingestion: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    dataset_dir = r"C:\Users\kunji\OneDrive\Pictures\Desktop\Major_Project\MP5-RAG-System\dataset"
    pipeline = IngestionPipeline(
        dataset_dir=dataset_dir,
        collection_name="policy_documents",
        chunk_output_dir=None,
    )
    pipeline.run()
