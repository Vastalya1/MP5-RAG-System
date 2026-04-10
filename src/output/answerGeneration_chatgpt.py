import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI


load_dotenv()


class AnswerGenerator:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Answer Generator with the OpenAI API."""
        openai_api_key =os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = AsyncOpenAI(api_key=openai_api_key)
        self.model_name = "gpt-4o-mini"

        self.ANSWER_PROMPT = """You are an expert assistant specialized in medical insurance policies.

A user has asked the following question:
"{rewritten_query}"

Below are the most relevant document chunks from the insurance policy, along with their section headings:

{chunks_text}

1. Extract the EXACT answer from the provided context.
2. Do NOT paraphrase or rephrase unless absolutely necessary.
3. Preserve exact numbers, limits, durations, and conditions.
4. Keep the answer as SHORT as possible.
5. Do NOT add explanations or extra information.
6. If the answer is not present, return: Not found
7. Do NOT hallucinate.
8. Read the document chunks properly and answer the questions.

9. IMPORTANT: Return PLAIN TEXT only. Do NOT use any markdown formatting such as:
   - No asterisks for bold (**text**)
   - No underscores for italics
   - No hash symbols for headings
   - No bullet points with dashes or asterisks
   - No horizontal rules (---)
   Just use plain sentences and paragraphs.

Present your answer in this format:

Answer:
[Your plain text answer here, simple and readable]

Justification:
Referenced from Section: [section_heading]"""

    def _format_chunks_for_prompt(self, chunks: List[Dict]) -> str:
        """Format chunks into a string for the prompt."""
        chunks_text = ""
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            chunks_text += f"\nChunk {i}:\n"
            chunks_text += f"Section: {metadata.get('section_heading', '')}\n"
            chunks_text += f"Text: {chunk.get('text', '')}\n"
            chunks_text += "-" * 80 + "\n"
        return chunks_text

    async def generate_answer(self, rewritten_query: str, reranked_chunks: List[Dict]) -> Dict:
        """
        Generate an answer using the rewritten query and reranked chunks.

        Args:
            rewritten_query: The query after being processed by QueryRewriter
            reranked_chunks: List of chunks after being processed by ChunkReranker

        Returns:
            Dict containing the generated answer and metadata
        """
        try:
            chunks_text = self._format_chunks_for_prompt(reranked_chunks)

            prompt = self.ANSWER_PROMPT.format(
                rewritten_query=rewritten_query,
                chunks_text=chunks_text,
            )

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert assistant specialized in medical insurance policies. "
                        "Provide clear, concise answers and always reference the relevant policy sections."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                top_p=0.95,
                max_tokens=500,
            )

            if response and response.choices:
                answer_content = response.choices[0].message.content
                if not answer_content:
                    raise Exception("Empty response from OpenAI")

                answer_text = answer_content.strip()

                answer_parts = answer_text.split("Justification:")
                main_answer = answer_parts[0].replace("Answer:", "").strip()

                justification = ""
                if len(answer_parts) > 1:
                    section_parts = answer_parts[1].split("Referenced from Section:")
                    if len(section_parts) > 1:
                        justification = "Referenced from Section: " + section_parts[1].strip()
                    else:
                        justification = answer_parts[1].strip()

                response_object = {
                    "answer": main_answer,
                    "justification": justification,
                    "source_chunks": [
                        {
                            "document": chunk.get("metadata", {}).get("document_name", ""),
                            "section": chunk.get("metadata", {}).get("section_heading", ""),
                            "text": chunk.get("text", "")[:200] + "...",
                        }
                        for chunk in reranked_chunks
                    ],
                    "metadata": {
                        "original_query": rewritten_query,
                        "num_chunks_used": len(reranked_chunks),
                        "model": self.model_name,
                    },
                }

                print(" Successfully generated answer")
                return response_object

            raise Exception("Empty response from OpenAI")

        except Exception as e:
            error_response = {
                "error": str(e),
                "answer": (
                    "I apologize, but I encountered an error while generating the answer. "
                    "Please try rephrasing your question."
                ),
                "justification": None,
                "source_chunks": [],
                "metadata": {
                    "error_type": type(e).__name__,
                    "original_query": rewritten_query,
                    "model": self.model_name,
                },
            }
            print(f"Error in answer generation: {str(e)}")
            return error_response


# # Example usage
# if __name__ == "__main__":
#     import asyncio
#     from src.queryRewriter.rewriting import QueryRewriter
#     from src.retriever.retrival import retrivalModel
#     from src.retriever.reranking import ChunkReranker
#
#     async def main():
#         # Initialize components
#         query_rewriter = QueryRewriter(os.getenv("MISTRAL_API_KEY"))
#         retriever = retrivalModel()
#         reranker = ChunkReranker(os.getenv("MISTRAL_API_KEY"))
#         answer_generator = AnswerGenerator()
#
#         # Example query
#         original_query = "What's covered for pregnancy?"
#
#         # Complete pipeline
#         try:
#             # 1. Rewrite query
#             rewritten_query = await query_rewriter.rewrite_query(original_query)
#             if not rewritten_query:
#                 raise Exception("Query rewriting failed")
#
#             # 2. Retrieve initial chunks
#             initial_chunks = retriever.retrive_Chunks(rewritten_query)
#
#             # 3. Rerank chunks
#             reranked_chunks = await reranker.rerank_chunks(rewritten_query, initial_chunks)
#
#             # 4. Generate answer
#             answer = await answer_generator.generate_answer(rewritten_query, reranked_chunks)
#
#             # Print results
#             print("\nGenerated Answer:")
#             print(answer)
#
#         except Exception as e:
#             print(f"Error in pipeline: {str(e)}")
#
#     asyncio.run(main())
