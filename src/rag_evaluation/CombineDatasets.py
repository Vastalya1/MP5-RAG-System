"""
Combine Golden Dataset and RAG Evaluation Results

This script combines golden_dataset.xlsx and rag_evaluation_results.json 
into a unified JSON format suitable for RAG evaluation metrics.
"""

import json
import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


def parse_dataset_chunks_md(md_path: str) -> Dict[str, str]:
    """
    Parse dataset_chunks.md to extract chunk content by chunk ID.
    
    Args:
        md_path: Path to the dataset_chunks.md file
        
    Returns:
        Dictionary mapping chunk_id to chunk content
    """
    chunks_dict = {}
    
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match chunk sections
    # Looks for "### Chunk: {chunk_id}" followed by content until "**Document Content:**" and then ```content```
    chunk_pattern = r'### Chunk: ([^\n]+)\n.*?(?=\*\*Document Content:\*\*)\*\*Document Content:\*\*\s*```\s*(.*?)```'
    
    matches = re.findall(chunk_pattern, content, re.DOTALL)
    
    for chunk_id, chunk_content in matches:
        chunk_id = chunk_id.strip()
        chunk_content = chunk_content.strip()
        chunks_dict[chunk_id] = chunk_content
    
    print(f"[Parser] Loaded {len(chunks_dict)} chunks from dataset_chunks.md")
    return chunks_dict


def get_chunk_content(chunk_id: str, chunks_dict: Dict[str, str]) -> str:
    """
    Get chunk content from the parsed chunks dictionary.
    
    Args:
        chunk_id: The chunk ID to look up
        chunks_dict: Dictionary mapping chunk_id to content
        
    Returns:
        The chunk content or empty string if not found
    """
    return chunks_dict.get(chunk_id, "")


def combine_datasets(
    excel_path: str,
    json_path: str,
    chunks_md_path: str,
    output_path: str,
    sheet_name: str = "dataset"
) -> List[Dict[str, Any]]:
    """
    Combine golden dataset and RAG evaluation results into unified format.
    
    Args:
        excel_path: Path to golden_dataset.xlsx
        json_path: Path to rag_evaluation_results.json
        chunks_md_path: Path to dataset_chunks.md
        output_path: Path for output JSON file
        sheet_name: Sheet name in Excel file
        
    Returns:
        List of combined result dictionaries
    """
    # Parse dataset_chunks.md for reference contexts
    print(f"\n[Combiner] Parsing dataset_chunks.md...")
    chunks_dict = parse_dataset_chunks_md(chunks_md_path)
    
    # Read golden dataset
    print(f"[Combiner] Reading golden dataset from {excel_path}...")
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    # Read RAG evaluation results
    print(f"[Combiner] Reading RAG results from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        rag_results = json.load(f)
    
    # Create a lookup dictionary for RAG results by question_id
    rag_results_dict = {r['question_id']: r for r in rag_results}
    
    # Combine the data
    combined_results = []
    
    for idx, row in df.iterrows():
        question_id = row['question_id']
        question = row['question']
        gold_answer = row['gold_answer']
        gold_chunks_raw = row['gold_chunks']
        
        # Parse gold_chunks (can be single value or comma-separated)
        if pd.isna(gold_chunks_raw):
            gold_chunk_ids = []
        elif isinstance(gold_chunks_raw, str):
            # Handle comma-separated or single value
            gold_chunk_ids = [c.strip() for c in gold_chunks_raw.split(',') if c.strip()]
        else:
            gold_chunk_ids = [str(gold_chunks_raw)]
        
        # Get reference contexts from dataset_chunks.md
        reference_contexts = []
        for chunk_id in gold_chunk_ids:
            chunk_content = get_chunk_content(chunk_id, chunks_dict)
            if chunk_content:
                reference_contexts.append(chunk_content)
            else:
                print(f"  Warning: Chunk '{chunk_id}' not found in dataset_chunks.md")
        
        # Get RAG results for this question
        rag_result = rag_results_dict.get(question_id, {})
        
        # Extract retrieved information
        retrieved_contexts = rag_result.get('chunk_content', [])
        retrieved_context_ids = rag_result.get('chunk_id', [])
        response = rag_result.get('answer', "")
        
        # Build the combined record
        combined_record = {
            "question_id": question_id,
            "user_input": question,
            "retrieved_contexts": retrieved_contexts,
            "reference_contexts": reference_contexts,
            "retrieved_context_ids": retrieved_context_ids,
            "reference_context_ids": gold_chunk_ids,
            "response": response,
            "reference": gold_answer
        }
        
        combined_results.append(combined_record)
    
    # Save to output JSON
    print(f"\n[Combiner] Writing combined results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    
    print(f"[Combiner] Successfully combined {len(combined_results)} records")
    return combined_results


def main():
    """Main entry point."""
    # Set up paths
    base_dir = Path(__file__).resolve().parent
    
    excel_path = base_dir / "golden_dataset.xlsx"
    json_path = base_dir / "Rag_evaluation_results/rag_evaluation_results_lexical_chatgpt.json" 
    chunks_md_path = base_dir / "dataset_chunks.md"
    output_path = base_dir / "combined_evaluation_dataset/combined_evaluation_dataset_lexical_chatgpt.json"
    
    # Verify files exist
    for path, name in [(excel_path, "golden_dataset.xlsx"), 
                       (json_path, "rag_evaluation_results_lexical_chatgpt.json"),
                       (chunks_md_path, "dataset_chunks.md")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at {path}")
    
    # Run combination
    results = combine_datasets(
        excel_path=str(excel_path),
        json_path=str(json_path),
        chunks_md_path=str(chunks_md_path),
        output_path=str(output_path),
        sheet_name="dataset"
    )
    
    print(f"\n[Main] Done! Output saved to {output_path}")
    print(f"[Main] Total records: {len(results)}")
    
    # Print sample output
    if results:
        print("\n[Main] Sample output (first record):")
        sample = results[0]
        print(f"  user_input: {sample['user_input'][:80]}...")
        print(f"  retrieved_contexts: {len(sample['retrieved_contexts'])} contexts")
        print(f"  reference_contexts: {len(sample['reference_contexts'])} contexts")
        print(f"  retrieved_context_ids: {sample['retrieved_context_ids']}")
        print(f"  reference_context_ids: {sample['reference_context_ids']}")
        print(f"  response: {sample['response'][:80]}..." if sample['response'] else "  response: (empty)")
        print(f"  reference: {str(sample['reference'])[:80]}..." if sample['reference'] else "  reference: (empty)")


if __name__ == "__main__":
    main()
