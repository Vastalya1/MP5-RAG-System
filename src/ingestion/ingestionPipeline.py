import os
from typing import Optional
from pathlib import Path
from .chunker import chunk_pdfs
from .embedder import DocumentEmbedder

class IngestionPipeline:
    def __init__(self, 
                 dataset_dir: str,
                 collection_name: str = "policy_documents",
                 chunk_output_dir: Optional[str] = None):
        """
        Initialize the ingestion pipeline
        
        Args:
            dataset_dir: Directory containing PDF files
            collection_name: Name for the ChromaDB collection
            chunk_output_dir: Optional directory to save JSON chunks
        """
        self.dataset_dir = Path(dataset_dir)
        self.chunk_output_dir = Path(chunk_output_dir) if chunk_output_dir else None
        self.collection_name = collection_name
        
        # Initialize embedder
        self.embedder = DocumentEmbedder(collection_name=collection_name)
        
    def run(self) -> None:
        """
        Run the complete ingestion pipeline:
        1. Chunk PDF documents
        2. Create embeddings
        3. Store in ChromaDB
        """
        print("🚀 Starting ingestion pipeline...")
        print(f"📂 Processing PDFs from: {self.dataset_dir}")
        
        # Create chunk output file path if specified
        chunk_output_file = None
        if self.chunk_output_dir:
            self.chunk_output_dir.mkdir(parents=True, exist_ok=True)
            chunk_output_file = str(self.chunk_output_dir / "chunks.json")
        
        try:
            # Step 1: Chunk the PDFs
            print("\n📄 Step 1: Chunking PDFs...")
            chunks = chunk_pdfs(
                input_folder=str(self.dataset_dir),
                output_file=chunk_output_file
            )
            print(f"✅ Chunking complete - Created {len(chunks)} chunks")
            
            # Step 2: Create embeddings and store in ChromaDB
            print("\n🔢 Step 2: Creating embeddings and storing in ChromaDB...")
            self.embedder.embed_documents(chunks)
            
            print("\n✨ Ingestion pipeline completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Error during ingestion: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    dataset_dir = r"D:\_official_\_MIT ADT_\_SEMESTER 7_\MP5\MP5-RAG-System\dataset"
    output_dir = None
    
    pipeline = IngestionPipeline(
        dataset_dir=dataset_dir,
        collection_name="policy_documents",
        chunk_output_dir=output_dir
    )
    
    pipeline.run()
