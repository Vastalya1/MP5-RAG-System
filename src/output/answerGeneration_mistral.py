from typing import List, Dict, Optional
import json
from mistralai import Mistral

class AnswerGenerator:
    def __init__(self, api_key: str):
        """Initialize the Answer Generator with Mistral API"""
        self.client = Mistral(api_key=api_key)
        self.model_name = "mistral-small"  # Using the same model as query rewriter
        
        self.ANSWER_PROMPT = """You are an expert assistant specialized in medical insurance policies.

A user has asked the following question:
"{rewritten_query}"

Below are the most relevant document chunks from the insurance policy, along with their section headings:

{chunks_text}

Instructions:

Carefully read all the provided chunks. Focus on the top 5 most relevant chunks if there are many.

Provide a clear, concise, and easy-to-understand answer for a common user, avoiding unnecessary technical terms.

Use medical and insurance terminology only when needed, and explain it in simple words if you do.

Justify your answer by referencing the chunk(s) used and their section headings.

If information is missing or unclear, explicitly say that instead of guessing.

Do not hallucinarte.

Present your answer in this format:

Answer:
[Your answer here, simple and readable]

Justification:

Referenced from Section: [section_heading]"""

    def _format_chunks_for_prompt(self, chunks: List[Dict]) -> str:
        """Format chunks into a string for the prompt"""
        chunks_text = ""
        for i, chunk in enumerate(chunks, 1):
            chunks_text += f"\nChunk {i}:\n"
            chunks_text += f"Section: {chunk['metadata']['section_heading']}\n"
            chunks_text += f"Text: {chunk['text']}\n"
            chunks_text += "-" * 80 + "\n"
        return chunks_text

    async def generate_answer(self, rewritten_query: str, reranked_chunks: List[Dict]) -> Dict:
        """
        Generate an answer using the rewritten query and reranked chunks
        
        Args:
            rewritten_query: The query after being processed by QueryRewriter
            reranked_chunks: List of chunks after being processed by ChunkReranker
            
        Returns:
            Dict containing the generated answer and metadata
        """
        try:
            # Format chunks for the prompt
            chunks_text = self._format_chunks_for_prompt(reranked_chunks)
            
            # Format the complete prompt
            prompt = self.ANSWER_PROMPT.format(
                rewritten_query=rewritten_query,
                chunks_text=chunks_text
            )
            
            # Create chat messages
            messages = [
                {"role": "system", "content": "You are an expert assistant specialized in medical insurance policies. Provide clear, concise answers and always reference the relevant policy sections."},
                {"role": "user", "content": prompt}
            ]

            # Get response from Mistral API
            response = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent answers
                top_p=0.95,
                max_tokens=500  # Longer responses for detailed answers
            )
            
            if response and response.choices:
                answer_text = response.choices[0].message.content.strip()
                
                # Parse the response into parts
                answer_parts = answer_text.split("Justification:")
                main_answer = answer_parts[0].replace("Answer:", "").strip()
                
                # Extract just the section references from justification
                justification = ""
                if len(answer_parts) > 1:
                    # Get only the "Referenced from Section:" part
                    section_parts = answer_parts[1].split("Referenced from Section:")
                    if len(section_parts) > 1:
                        justification = "Referenced from Section: " + section_parts[1].strip()
                    else:
                        justification = answer_parts[1].strip()
                
                # Create response object
                response_object = {
                    "answer": main_answer,
                    "justification": justification,
                    "source_chunks": [
                        {
                            "document": chunk["metadata"]["document_name"],
                            "section": chunk["metadata"]["section_heading"],
                            "text": chunk["text"][:200] + "..."  # Truncated preview
                        }
                        for chunk in reranked_chunks[:5]  # Include top 5 chunks
                    ],
                    "metadata": {
                        "original_query": rewritten_query,
                        "num_chunks_used": len(reranked_chunks)
                    }
                }
                
                print("✨ Successfully generated answer")
                return response_object
                
            else:
                raise Exception("Empty response from Gemini")
                
        except Exception as e:
            error_response = {
                "error": str(e),
                "answer": "I apologize, but I encountered an error while generating the answer. Please try rephrasing your question.",
                "justification": None,
                "source_chunks": [],
                "metadata": {
                    "error_type": type(e).__name__,
                    "original_query": rewritten_query
                }
            }
            print(f"Error in answer generation: {str(e)}")
            return error_response


# # Example usage
# if __name__ == "__main__":
#     import asyncio
#     from src.queryRewriter.rewriting import QueryRewriter
#     from src.retriever.retrival import retrivalModel
#     from src.retriever.reranking import ChunkReranker
    
#     async def main():
#         # Initialize components
#         API_KEY = "your-api-key-here"
#         query_rewriter = QueryRewriter(API_KEY)
#         retriever = retrivalModel()
#         reranker = ChunkReranker(API_KEY)
#         answer_generator = AnswerGenerator(API_KEY)
        
#         # Example query
#         original_query = "What's covered for pregnancy?"
        
#         # Complete pipeline
#         try:
#             # 1. Rewrite query
#             rewritten_query = await query_rewriter.rewrite_query(original_query)
#             if not rewritten_query:
#                 raise Exception("Query rewriting failed")
                
#             # 2. Retrieve initial chunks
#             initial_chunks = retriever.retrive_Chunks(rewritten_query)
            
#             # 3. Rerank chunks
#             reranked_chunks = await reranker.rerank_chunks(rewritten_query, initial_chunks)
            
#             # 4. Generate answer
#             answer = await answer_generator.generate_answer(rewritten_query, reranked_chunks)
            
#             # Print results
#             print("\nGenerated Answer:")
#             print(json.dumps(answer, indent=2))
            
#         except Exception as e:
#             print(f"Error in pipeline: {str(e)}")
    
#     asyncio.run(main())
