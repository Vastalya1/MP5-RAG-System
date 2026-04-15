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
from typing import Any, Dict, List
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).resolve().parent.parent))

from output.answerGeneration_Chatgpt import AnswerGenerator
from queryRewriter.rewriting_Chatgpt import QueryRewriter
from retriever.reranking_Chatgpt import ChunkReranker
from retriever.retrival import retrivalModel


class RAGTestPipeline:
    def __init__(self, api_key: str):
        self.api_key = api_key
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
        policy_id: str = None,
    ) -> Dict[str, Any]:
        try:
            print(f"\n[Processing Q{question_id}] {question[:50]}...")
            print("  Step 1: Rewriting query...")
            rewritten_query = await self.rewriter.rewrite_query(question)
            if not rewritten_query:
                rewritten_query = question

            print("  Step 2: Retrieving chunks...")
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

            print("  Step 3: Reranking chunks...")
            reranked_chunks = await self.reranker.rerank_chunks(rewritten_query, chunks, top_k=5)

            print("  Step 4: Generating answer...")
            answer_result = await self.answer_generator.generate_answer(rewritten_query, reranked_chunks)

            chunk_ids = []
            chunk_contents = []
            policy_ids = set()

            for chunk in reranked_chunks:
                chunk_ids.append(chunk.get("chunk_id", ""))
                chunk_contents.append(chunk.get("text", ""))
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
    ) -> List[Dict[str, Any]]:
        print(f"\n[RAGTestPipeline] Reading questions from {excel_path}...")
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        question_col = None
        for col in ["question", "Question", "questions", "Questions", "query", "Query"]:
            if col in df.columns:
                question_col = col
                break

        if question_col is None:
            question_col = df.columns[0]
            print(f"[RAGTestPipeline] Using column '{question_col}' as question column")

        id_col = None
        for col in ["question_id", "Question_ID", "id", "ID", "qid", "QID"]:
            if col in df.columns:
                id_col = col
                break

        questions = df[question_col].tolist()
        num_questions = len(questions)
        print(f"[RAGTestPipeline] Found {num_questions} questions to process")

        policy_col = None
        for col in ["policy_id", "Policy_ID", "policy", "document_name"]:
            if col in df.columns:
                policy_col = col
                break

        if policy_col:
            print(f"[RAGTestPipeline] Using '{policy_col}' column for metadata filtering")

        results = []
        for idx, question in enumerate(questions):
            q_id = df[id_col].iloc[idx] if id_col else idx + 1
            policy_id = df[policy_col].iloc[idx] if policy_col else None
            result = await self.process_question(q_id, question, collection_name, policy_id)
            results.append(result)
            print(f"  Completed {idx + 1}/{num_questions}")

        if output_path is None:
            output_dir = Path(__file__).resolve().parent
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"rag_results_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n[RAGTestPipeline] Results saved to {output_path}")
        return results


async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    base_dir = Path(__file__).resolve().parent
    excel_path = base_dir / "golden_dataset.xlsx"
    output_path = base_dir / "rag_evaluation_results.json"

    pipeline = RAGTestPipeline(api_key)
    results = await pipeline.run_pipeline(
        excel_path=str(excel_path),
        sheet_name="dataset",
        output_path=str(output_path),
        collection_name="dataset",
    )

    print(f"\n[RAGTestPipeline] Processed {len(results)} questions")
    print(f"[RAGTestPipeline] Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
