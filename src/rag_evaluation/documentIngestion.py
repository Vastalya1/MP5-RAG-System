"""
Document Ingestion Script for RAG Evaluation

This script processes PDF documents in the sampleDataset folder,
chunks them, creates embeddings, and stores them in ChromaDB.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ingestion.ingestionPipeline import IngestionPipeline
from ingestion.chunker import chunk_pdfs
from ingestion.embedder import DocumentEmbedder


def ingest_sample_dataset(
    collection_name: str = "dataset",
    save_chunks_json: bool = False
) -> None:
    """
    Ingest all PDF documents from the sampleDataset folder into ChromaDB.
    
    Args:
        collection_name: Name of the ChromaDB collection to store chunks
        save_chunks_json: Whether to save chunks to a JSON file for debugging
    """
    # Get the path to sampleDataset folder
    current_dir = Path(__file__).resolve().parent
    sample_dataset_dir = current_dir / "sampleDataset"
    
    if not sample_dataset_dir.exists():
        print(f"Error: sampleDataset folder not found at {sample_dataset_dir}")
        return
    
    # List PDF files in the directory
    pdf_files = list(sample_dataset_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {sample_dataset_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files in sampleDataset:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    # Set up chunk output directory if needed
    chunk_output_dir = None
    if save_chunks_json:
        chunk_output_dir = str(current_dir / "output")
    
    # Create and run the ingestion pipeline
    pipeline = IngestionPipeline(
        dataset_dir=str(sample_dataset_dir),
        collection_name=collection_name,
        chunk_output_dir=chunk_output_dir,
    )
    
    print(f"\nStarting ingestion pipeline...")
    print(f"Collection name: {collection_name}")
    print("-" * 50)
    
    pipeline.run()
    
    print("-" * 50)
    print("Document ingestion completed!")


def ingest_specific_files(
    file_paths: list,
    collection_name: str = "dataset"
) -> None:
    """
    Ingest specific PDF files into ChromaDB.
    
    Args:
        file_paths: List of paths to PDF files to ingest
        collection_name: Name of the ChromaDB collection to store chunks
    """
    current_dir = Path(__file__).resolve().parent
    sample_dataset_dir = current_dir / "sampleDataset"
    
    pipeline = IngestionPipeline(
        dataset_dir=str(sample_dataset_dir),
        collection_name=collection_name,
        file_paths=file_paths,
    )
    
    pipeline.run()


if __name__ == "__main__":
    # Run the ingestion for sampleDataset folder
    # Collection name: dataset
    ingest_sample_dataset(
        collection_name="dataset",
        save_chunks_json=False  # Set to True if you want to save chunks to JSON
    )
