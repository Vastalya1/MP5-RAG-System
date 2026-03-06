"""
ChromaDB to Markdown Exporter

This script reads all chunks from the 'dataset' collection in ChromaDB
and exports them to a markdown file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb

# Load environment variables
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)


def export_chunks_to_markdown(
    collection_name: str = "dataset",
    output_file: str = None
) -> None:
    """
    Export all chunks from ChromaDB collection to a markdown file.
    
    Args:
        collection_name: Name of the ChromaDB collection
        output_file: Path to the output markdown file
    """
    # Connect to ChromaDB Cloud
    client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_CLOUD_API_KEY"),
        tenant='a92961b0-ea65-4a82-a7ad-321a4baaaa60',
        database='Major-Project'
    )
    
    # Get the collection
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error: Could not get collection '{collection_name}': {e}")
        return
    
    # Get all documents from the collection
    print(f"Fetching documents from collection: {collection_name}")
    
    # Get total count first
    total_count = collection.count()
    print(f"Total documents in collection: {total_count}")
    
    # Fetch in batches due to ChromaDB Cloud quota limit
    batch_size = 300
    all_ids = []
    all_documents = []
    all_metadatas = []
    
    for offset in range(0, total_count, batch_size):
        print(f"Fetching batch {offset // batch_size + 1} (offset: {offset})...")
        results = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset
        )
        all_ids.extend(results.get("ids", []))
        all_documents.extend(results.get("documents", []))
        all_metadatas.extend(results.get("metadatas", []))
    
    ids = all_ids
    documents = all_documents
    metadatas = all_metadatas
    
    total_chunks = len(ids)
    print(f"Found {total_chunks} chunks in collection")
    
    if total_chunks == 0:
        print("No chunks to export.")
        return
    
    # Set default output path
    if output_file is None:
        output_dir = Path(__file__).resolve().parent
        output_file = str(output_dir / "dataset_chunks.md")
    
    # Group chunks by document name for better organization
    chunks_by_doc = {}
    for i, chunk_id in enumerate(ids):
        metadata = metadatas[i] if i < len(metadatas) else {}
        document = documents[i] if i < len(documents) else ""
        
        doc_name = metadata.get("document_name", "unknown_document")
        section_heading = metadata.get("section_heading", "General")
        
        if doc_name not in chunks_by_doc:
            chunks_by_doc[doc_name] = []
        
        chunks_by_doc[doc_name].append({
            "chunk_id": chunk_id,
            "section_heading": section_heading,
            "document": document
        })
    
    # Write to markdown file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Dataset Chunks Export\n\n")
        f.write(f"**Total Chunks:** {total_chunks}\n\n")
        f.write(f"**Documents:** {len(chunks_by_doc)}\n\n")
        f.write("---\n\n")
        
        for doc_name, chunks in chunks_by_doc.items():
            f.write(f"## {doc_name}\n\n")
            f.write(f"**Chunks in this document:** {len(chunks)}\n\n")
            
            for chunk in chunks:
                f.write(f"### Chunk: {chunk['chunk_id']}\n\n")
                f.write(f"| Field | Value |\n")
                f.write(f"|-------|-------|\n")
                f.write(f"| **Document Name** | {doc_name} |\n")
                f.write(f"| **Section Heading** | {chunk['section_heading']} |\n")
                f.write(f"| **Chunk ID** | {chunk['chunk_id']} |\n\n")
                f.write(f"**Document Content:**\n\n")
                f.write(f"```\n{chunk['document']}\n```\n\n")
                f.write("---\n\n")
    
    print(f"Successfully exported {total_chunks} chunks to: {output_file}")


def export_chunks_to_csv_style_markdown(
    collection_name: str = "dataset",
    output_file: str = None
) -> None:
    """
    Export chunks in a simpler table format.
    """
    client = chromadb.CloudClient(
        api_key=os.getenv("CHROMA_CLOUD_API_KEY"),
        tenant='a92961b0-ea65-4a82-a7ad-321a4baaaa60',
        database='Major-Project'
    )
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error: Could not get collection '{collection_name}': {e}")
        return
    
    total_count = collection.count()
    print(f"Total documents: {total_count}")
    
    # Fetch in batches due to ChromaDB Cloud quota limit
    batch_size = 300
    all_ids = []
    all_documents = []
    all_metadatas = []
    
    for offset in range(0, total_count, batch_size):
        results = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset
        )
        all_ids.extend(results.get("ids", []))
        all_documents.extend(results.get("documents", []))
        all_metadatas.extend(results.get("metadatas", []))
    
    ids = all_ids
    documents = all_documents
    metadatas = all_metadatas
    
    total_chunks = len(ids)
    print(f"Found {total_chunks} chunks")
    
    if output_file is None:
        output_dir = Path(__file__).resolve().parent
        output_file = str(output_dir / "dataset_chunks_table.md")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Dataset Chunks\n\n")
        f.write(f"Total: {total_chunks} chunks\n\n")
        
        for i, chunk_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            document = documents[i] if i < len(documents) else ""
            
            doc_name = metadata.get("document_name", "unknown_document")
            section_heading = metadata.get("section_heading", "General")
            
            f.write(f"## {i + 1}. {chunk_id}\n\n")
            f.write(f"- **Document Name:** {doc_name}\n")
            f.write(f"- **Section Heading:** {section_heading}\n")
            f.write(f"- **Chunk ID:** {chunk_id}\n")
            f.write(f"- **Document:**\n\n")
            f.write(f"> {document}\n\n")
            f.write("---\n\n")
    
    print(f"Exported to: {output_file}")


if __name__ == "__main__":
    # Export chunks to markdown (grouped by document)
    export_chunks_to_markdown(
        collection_name="dataset",
        output_file=None  # Will save to dataset_chunks.md
    )
