"""
RAG Test Pipeline

This script reads questions from golden_dataset.xlsx and runs them through
the RAG pipeline using BM25 lexical retrieval and OpenAI for query rewriting,
reranking, and answer generation, storing results in a JSON file.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from output.answerGeneration_chatgpt import AnswerGenerator
from queryRewriter.rewriting_chatgpt import QueryRewriter
from retriever.reranking_chatgpt import ChunkReranker
from retriever.retrival_lexical import retrivalModel


def parse_question_numbers(value: Optional[str]) -> Optional[Set[int]]:
    """
    Parse question numbers from a comma-separated string with optional ranges.

    Example:
        "1,3,5-7" -> {1, 3, 5, 6, 7}
    """
    if not value:
        return None

    question_numbers: Set[int] = set()
    parts = [part.strip() for part in value.split(",") if part.strip()]

    for part in parts:
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if start > end:
                raise ValueError(
                    f"Invalid range '{part}'. Start must be less than or equal to end."
                )
            question_numbers.update(range(start, end + 1))
        else:
            question_numbers.add(int(part))

    if any(number <= 0 for number in question_numbers):
        raise ValueError("Question numbers must be positive integers.")

    return question_numbers


class RAGTestPipeline:
    """
    Test pipeline to evaluate RAG system performance using a golden dataset.
    """

    def __init__(self, api_key: str):
        """
        Initialize the RAG test pipeline components.

        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        self.sleep_between_questions = float(os.getenv("RAG_TEST_SLEEP_SECONDS", "1.5"))

        # Initialize pipeline components
        print("[RAGTestPipeline] Initializing OpenAI components...")
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
        policy_id: str = None,
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
            print("  Step 1: Rewriting query...")
            rewritten_query = await self.rewriter.rewrite_query(question)
            if not rewritten_query:
                rewritten_query = question

            # Step 2: Retrieval (with document filter if policy_id provided)
            print("  Step 2: Retrieving chunks with lexical BM25...")
            chunks = self.retriever.retrive_Chunks(
                rewritten_query,
                collection_name=collection_name,
                top_k=15,
                document_filter=policy_id,
            )

            if not chunks:
                return {
                    "question_id": question_id,
                    "question": question,
                    "answer": "No relevant policy content found for this question.",
                    "policy_id": [],
                    "chunk_id": [],
                    "chunk_content": [],
                }

            # Step 3: Reranking
            print("  Step 3: Reranking chunks...")
            reranked_chunks = await self.reranker.rerank_chunks(rewritten_query, chunks)

            # Step 4: Answer Generation
            print("  Step 4: Generating answer...")
            answer_result = await self.answer_generator.generate_answer(
                rewritten_query,
                reranked_chunks,
            )

            # Extract chunk information
            chunk_ids = []
            chunk_contents = []
            policy_ids = set()

            for chunk in reranked_chunks:
                chunk_ids.append(chunk.get("chunk_id", ""))
                chunk_contents.append(chunk.get("text", ""))

                # Extract policy_id from metadata
                metadata = chunk.get("metadata", {})
                doc_name = metadata.get("document_name", "")
                if doc_name:
                    policy_ids.add(doc_name)

            return {
                "question_id": question_id,
                "question": question,
                "answer": answer_result.get("answer", ""),
                "policy_id": list(policy_ids),
                "chunk_id": chunk_ids,
                "chunk_content": chunk_contents,
            }

        except Exception as e:
            print(f"  Error processing question {question_id}: {str(e)}")
            return {
                "question_id": question_id,
                "question": question,
                "answer": f"Error: {str(e)}",
                "policy_id": [],
                "chunk_id": [],
                "chunk_content": [],
            }

    async def run_pipeline(
        self,
        excel_path: str,
        sheet_name: str = "dataset",
        output_path: str = None,
        collection_name: str = "dataset",
        question_numbers: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run the RAG pipeline on all questions from the Excel file.

        Args:
            excel_path: Path to the golden_dataset.xlsx file
            sheet_name: Name of the sheet containing questions
            output_path: Path for the output JSON file
            collection_name: ChromaDB collection to query
            question_numbers: Optional 1-based question positions to process

        Returns:
            List of result dictionaries
        """
        # Read the Excel file
        print(f"\n[RAGTestPipeline] Reading questions from {excel_path}...")
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        # Check for question column (common column names)
        question_col = None
        for col in ["question", "Question", "questions", "Questions", "query", "Query"]:
            if col in df.columns:
                question_col = col
                break

        if question_col is None:
            # If no standard column name, use the first column
            question_col = df.columns[0]
            print(f"[RAGTestPipeline] Using column '{question_col}' as question column")

        # Check for question_id column
        id_col = None
        for col in ["question_id", "Question_ID", "id", "ID", "qid", "QID"]:
            if col in df.columns:
                id_col = col
                break

        questions = df[question_col].tolist()
        num_questions = len(questions)
        print(f"[RAGTestPipeline] Found {num_questions} questions to process")

        if question_numbers:
            valid_question_numbers = {
                number for number in question_numbers if 1 <= number <= num_questions
            }
            skipped_numbers = sorted(question_numbers - valid_question_numbers)

            if skipped_numbers:
                print(
                    f"[RAGTestPipeline] Skipping out-of-range question numbers: {skipped_numbers}"
                )

            if not valid_question_numbers:
                raise ValueError("No valid question numbers were provided.")

            print(
                f"[RAGTestPipeline] Running only question numbers: "
                f"{sorted(valid_question_numbers)}"
            )
            question_numbers = valid_question_numbers

        # Check for policy_id column for metadata filtering
        policy_col = None
        for col in ["policy_id", "Policy_ID", "policy", "document_name"]:
            if col in df.columns:
                policy_col = col
                break

        if policy_col:
            print(f"[RAGTestPipeline] Using '{policy_col}' column for metadata filtering")

        # Process each question
        results = []
        for idx, question in enumerate(questions):
            question_number = idx + 1

            if question_numbers and question_number not in question_numbers:
                continue

            # Use question_id from file or generate one
            if id_col:
                q_id = df[id_col].iloc[idx]
            else:
                q_id = question_number

            # Get policy_id for metadata filtering
            policy_id = None
            if policy_col:
                policy_id = df[policy_col].iloc[idx]

            result = await self.process_question(q_id, question, collection_name, policy_id)
            results.append(result)
            print(f"  Completed {idx + 1}/{num_questions}")

            if idx < num_questions - 1 and self.sleep_between_questions > 0:
                print(
                    f"  Sleeping for {self.sleep_between_questions} seconds to avoid rate limits..."
                )
                await asyncio.sleep(self.sleep_between_questions)

        # Save results to JSON
        if output_path is None:
            output_dir = Path(__file__).resolve().parent
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"rag_results_lexical_chatgpt_{timestamp}.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n[RAGTestPipeline] Results saved to {output_path}")
        return results


async def main():
    """Main entry point for the RAG test pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the OpenAI lexical RAG test pipeline on all or selected questions."
    )
    parser.add_argument(
        "--question-numbers",
        type=str,
        default=None,
        help="Comma-separated 1-based question numbers or ranges, e.g. 1,3,5-7",
    )
    args = parser.parse_args()

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY_GEN")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Set up paths
    base_dir = Path(__file__).resolve().parent
    excel_path = base_dir / "golden_dataset.xlsx"
    output_path = base_dir / "Rag_evaluation_results/rag_evaluation_results_lexical_chatgpt2.json"

    # Initialize and run pipeline
    pipeline = RAGTestPipeline(api_key)
    results = await pipeline.run_pipeline(
        excel_path=str(excel_path),
        sheet_name="dataset",
        output_path=str(output_path),
        collection_name="dataset",
        question_numbers=parse_question_numbers(args.question_numbers),
    )

    print(f"\n[RAGTestPipeline] Processed {len(results)} questions")
    print(f"[RAGTestPipeline] Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
