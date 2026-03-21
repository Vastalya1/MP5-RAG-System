from typing import Optional, List
from pathlib import Path
from .chunker import chunk_pdfs, chunk_pdf_files, DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP
from .embedder import DocumentEmbedder


class IngestionPipeline:
    def __init__(
        self,
        dataset_dir: str,
        collection_name: str = "dataset",
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

    def _validated_file_paths(self) -> List[str]:
        valid_paths: List[str] = []
        for file_path in self.file_paths or []:
            path = Path(file_path)
            if not path.exists():
                print(f"Skipping missing file: {file_path}")
                continue
            if path.suffix.lower() != ".pdf":
                print(f"Skipping non-PDF file: {file_path}")
                continue
            valid_paths.append(str(path))
        return valid_paths

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
        print(
            f"Chunking config - max_tokens: {DEFAULT_MAX_TOKENS}, overlap: {int(DEFAULT_OVERLAP * 100)}%"
        )

        chunk_output_file = None
        if self.chunk_output_dir:
            self.chunk_output_dir.mkdir(parents=True, exist_ok=True)
            chunk_output_file = str(self.chunk_output_dir / "chunks.json")

        try:
            print("Step 1: Chunking PDFs...")
            if self.file_paths:
                valid_file_paths = self._validated_file_paths()
                if not valid_file_paths:
                    print("No valid PDF files to process.")
                    return
                chunks = chunk_pdf_files(
                    file_paths=valid_file_paths,
                    output_file=chunk_output_file,
                )
            else:
                chunks = chunk_pdfs(
                    input_folder=str(self.dataset_dir),
                    output_file=chunk_output_file,
                )
            print(f"Chunking complete - Created {len(chunks)} chunks")
            if not chunks:
                print("No chunks generated. Skipping embedding step.")
                return

            print("Step 2: Creating embeddings and storing in ChromaDB...")
            self.embedder.embed_documents(chunks)

            print("Ingestion pipeline completed successfully.")
        except Exception as e:
            print(f"Error during ingestion: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    dataset_dir = r"D:\_official_\_MIT ADT_\_SEMESTER 7_\MP5\MP5-RAG-System\dataset"
    pipeline = IngestionPipeline(
        dataset_dir=dataset_dir,
        collection_name="dataset",
        chunk_output_dir=None,
    )
    pipeline.run()
