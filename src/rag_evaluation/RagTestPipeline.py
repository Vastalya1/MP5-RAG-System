"""
RAG Test Pipeline

This script reads questions from golden_dataset.xlsx and runs them through 
the RAG pipeline, storing results in a JSON file.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from queryRewriter.rewriting import QueryRewriter
from retriever.retrival import retrivalModel
from retriever.reranking_mistral import ChunkReranker
from output.answerGeneration_mistral import AnswerGenerator


class RAGTestPipeline:
    """
    Test pipeline to evaluate RAG system performance using a golden dataset.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the RAG test pipeline components.
        
        Args:
            api_key: Mistral API key
        """
        self.api_key = api_key
        
        # Initialize pipeline components
        print("[RAGTestPipeline] Initializing components...")
        self.rewriter = QueryRewriter(api_key)
        self.retriever = retrivalModel()
        self.reranker = ChunkReranker(api_key)
        self.answer_generator = AnswerGenerator(api_key)
        print("[RAGTestPipeline] All components initialized successfully")
    
    async def process_question(
        self, 
        question_id: int, 
        question: str,
        collection_name: str = "dataset",
        policy_id: str = None
    ) -> Dict[str, Any]:
        """
        Process a single question through the RAG pipeline.
        
        Args:
            question_id: The ID of the question
            question: The question text
            collection_name: ChromaDB collection to query
            policy_id: The policy document name to filter retrieval by
            
        Returns:
            Dict containing the results
        """
        try:
            print(f"\n[Processing Q{question_id}] {question[:50]}...")
            
            # Step 1: Query Rewriting
            print(f"  Step 1: Rewriting query...")
            rewritten_query = await self.rewriter.rewrite_query(question)
            if not rewritten_query:
                rewritten_query = question
            
            # Step 2: Retrieval (with document filter if policy_id provided)
            print(f"  Step 2: Retrieving chunks...")
            chunks = self.retriever.retrive_Chunks(
                rewritten_query, 
                collection_name=collection_name,
                top_k=15,
                document_filter=policy_id
            )
            
            if not chunks:
                return {
                    "question_id": question_id,
                    "question": question,
                    "answer": "No relevant policy content found for this question.",
                    "policy_id": [],
                    "chunk_id": [],
                    "chunk_content": []
                }
            
            # Step 3: Reranking
            print(f"  Step 3: Reranking chunks...")
            reranked_chunks = await self.reranker.rerank_chunks(rewritten_query, chunks, top_k=5)
            
            # Step 4: Answer Generation
            print(f"  Step 4: Generating answer...")
            answer_result = await self.answer_generator.generate_answer(rewritten_query, reranked_chunks)
            
            # Extract chunk information
            chunk_ids = []
            chunk_contents = []
            policy_ids = set()
            
            for chunk in reranked_chunks:
                chunk_ids.append(chunk.get('chunk_id', ''))
                chunk_contents.append(chunk.get('text', ''))
                
                # Extract policy_id from metadata
                metadata = chunk.get('metadata', {})
                doc_name = metadata.get('document_name', '')
                if doc_name:
                    policy_ids.add(doc_name)
            
            return {
                "question_id": question_id,
                "question": question,
                "answer": answer_result.get("answer", ""),
                "policy_id": list(policy_ids),
                "chunk_id": chunk_ids,
                "chunk_content": chunk_contents
            }
            
        except Exception as e:
            print(f"  Error processing question {question_id}: {str(e)}")
            return {
                "question_id": question_id,
                "question": question,
                "answer": f"Error: {str(e)}",
                "policy_id": [],
                "chunk_id": [],
                "chunk_content": []
            }
    
    async def run_pipeline(
        self, 
        excel_path: str, 
        sheet_name: str = "dataset",
        output_path: str = None,
        collection_name: str = "dataset"
    ) -> List[Dict[str, Any]]:
        """
        Run the RAG pipeline on all questions from the Excel file.
        
        Args:
            excel_path: Path to the golden_dataset.xlsx file
            sheet_name: Name of the sheet containing questions
            output_path: Path for the output JSON file
            collection_name: ChromaDB collection to query
            
        Returns:
            List of result dictionaries
        """
        # Read the Excel file
        print(f"\n[RAGTestPipeline] Reading questions from {excel_path}...")
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # Check for question column (common column names)
        question_col = None
        for col in ['question', 'Question', 'questions', 'Questions', 'query', 'Query']:
            if col in df.columns:
                question_col = col
                break
        
        if question_col is None:
            # If no standard column name, use the first column
            question_col = df.columns[0]
            print(f"[RAGTestPipeline] Using column '{question_col}' as question column")
        
        # Check for question_id column
        id_col = None
        for col in ['question_id', 'Question_ID', 'id', 'ID', 'qid', 'QID']:
            if col in df.columns:
                id_col = col
                break
        
        questions = df[question_col].tolist()
        num_questions = len(questions)
        print(f"[RAGTestPipeline] Found {num_questions} questions to process")
        
        # Check for policy_id column for metadata filtering
        policy_col = None
        for col in ['policy_id', 'Policy_ID', 'policy', 'document_name']:
            if col in df.columns:
                policy_col = col
                break
        
        if policy_col:
            print(f"[RAGTestPipeline] Using '{policy_col}' column for metadata filtering")
        
        # Process each question
        results = []
        for idx, question in enumerate(questions):
            # Use question_id from file or generate one
            if id_col:
                q_id = df[id_col].iloc[idx]
            else:
                q_id = idx + 1
            
            # Get policy_id for metadata filtering
            policy_id = None
            if policy_col:
                policy_id = df[policy_col].iloc[idx]
            
            result = await self.process_question(q_id, question, collection_name, policy_id)
            results.append(result)
            print(f"  Completed {idx + 1}/{num_questions}")
        
        # Save results to JSON
        if output_path is None:
            output_dir = Path(__file__).resolve().parent
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"rag_results_{timestamp}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[RAGTestPipeline] Results saved to {output_path}")
        return results


async def main():
    """Main entry point for the RAG test pipeline."""
    # Get API key from environment
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    
    # Set up paths
    base_dir = Path(__file__).resolve().parent
    excel_path = base_dir / "golden_dataset.xlsx"
    output_path = base_dir / "rag_evaluation_results.json"
    
    # Initialize and run pipeline
    pipeline = RAGTestPipeline(api_key)
    results = await pipeline.run_pipeline(
        excel_path=str(excel_path),
        sheet_name="dataset",
        output_path=str(output_path),
        collection_name="dataset"
    )
    
    print(f"\n[RAGTestPipeline] Processed {len(results)} questions")
    print(f"[RAGTestPipeline] Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
