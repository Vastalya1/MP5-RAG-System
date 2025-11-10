import asyncio
from typing import List, Dict, Any
import json
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness
)
from ragas import evaluate
from ragas.embeddings import SentenceTransformersEmbeddings

# from ragas.llms import GoogleLLM

from datasets import Dataset
# Import all RAG pipeline components
from ingestion.ingestionPipeline import IngestionPipeline
from queryRewriter.rewriting import QueryRewriter
from retriever.retrival import retrivalModel
from retriever.reranking_mistral import ChunkReranker  # Using Mistral version
from output.answerGeneration_mistral import AnswerGenerator

# Initialize components with API keys
MISTRAL_API_KEY = "IQEYxF0lNL2msXCKTg1ryDz9v3tSiQ59"
GEMINI_API_KEY = "AIzaSyBLJ_eYaCBQ6TY4RUGf_gelHyU1H4pPw1g"

# Configure Gemini for evaluation
# genai.configure(api_key=GEMINI_API_KEY)
# llm = GoogleLLM(model_name="gemini-pro")

# Initialize RAG components
query_rewriter = QueryRewriter(MISTRAL_API_KEY)
retriever = retrivalModel()
reranker = ChunkReranker(MISTRAL_API_KEY)
answer_generator = AnswerGenerator(MISTRAL_API_KEY)

# Initialize embeddings
embedding_model = SentenceTransformersEmbeddings(model_name="intfloat/e5-base-v2")

queries = [
    "My father fell from stairs and broke his hip. He's 68 years old. The policy was taken last week only. Emergency case. Will insurance pay?",
    "Can I claim for my spectacles? My eye power is -6.5 and doctor says I need new glasses?",
    "Wife is pregnant, 7 months. Do you cover delivery charges? We have Imperial Plus plan for 14 months now.",
    "Can I get treatment in ayurvedic hospital? My back pain needs ayurvedic therapy.",
    "Policy document mentions cumulative bonus. What is that? How do I get it?",
    "I slipped in hotel bathroom during my Goa trip and fractured my arm. Got admitted for 2 days. Policy is 5 months old. Will you cover the hospital bills?",
    "Got dengue fever while on business trip to Mumbai. Hospitalized for 4 days. Policy purchased 3 weeks ago. Covered?",
    "Died in car accident during road trip. Policy has INR 5 lakh personal accident cover. Will nominee get payment?",
    "My mother accompanying me got sick during trip. She's not insured under this policy. Can I claim her medical expenses?",
    "I have diabetes and blood pressure. Took medicine along. During Jaipur trip, sugar went very high, admitted in ICU for 2 days. Policy is 8 months old. What happens?"
]

ground_truths = [
    # 1
    "Accidental injuries are covered immediately without waiting periods. Hip fracture due to fall is an accident (sudden, unforeseen, involuntary event caused by external, visible and violent means). Emergency hospitalization for fracture treatment is covered under In-patient Hospitalization Treatment.",

    # 2
    "Spectacles and contact lenses are explicitly excluded from coverage (Specific Exclusion 4). The policy only covers treatment for refractive error ≥7.5 dioptres after 24-month waiting period, and only when requiring medical intervention, not spectacles. Decision: REJECTED.",

    # 3
    "Maternity expenses including normal delivery, complicated deliveries, and caesarean sections are permanently excluded (Code-Excl18). Only ectopic pregnancy is covered as an exception. Duration of policy or plan type does not affect this exclusion. Decision: REJECTED.",

    # 4
    "Ayurvedic/Homeopathic Hospitalization Expenses covered for minimum 24-hour admission in government ayurvedic hospital or government-recognized/QCI/NABH-accredited institute (Section C, Part A, I-8). Covers room, nursing, consultation, medicines, ayurvedic treatment procedures. Must be medically necessary and require inpatient admission. Subject to waiting periods and exclusions. Decision: APPROVED if meets criteria.",

    # 5
    "Cumulative Bonus (Domestic Cover only): If policy renewed without break and NO claim made in preceding year, Sum Insured increases by 20% of base Sum Insured annually. Maximum cumulative increase: 100% of base Sum Insured. If claim made in any year with bonus, bonus reduces by 20% in next renewal (base Sum Insured preserved). Bonus does not reduce sum insured when used. Not a claim response - benefit explanation.",

    # 6
    "Accidental injury requiring emergency hospitalization is covered under Base Cover 1 - Emergency Accidental Hospitalization. Covers in-patient treatment, X-rays, diagnostic tests during hospitalization. The 30-day waiting period does not apply to accidents. Policy is 5 months old, so waiting period already passed. Decision: APPROVED subject to reasonable and customary charges and deductible.",

    # 7
    "Dengue is an illness/disease, not an accident. Base cover only includes Emergency Accidental Hospitalization, which covers accidents only. For illness coverage, Endorsement no.1 - Emergency Medical Expenses (Illness/Disease) must be opted and paid for separately. Even if opted, policy is only 3 weeks old, within 30-day waiting period for illness (General Exclusion 4). Decision: REJECTED.",

    # 8
    "Accidental Death is covered under Base Cover 3a - Personal Accident Covers. Accident during trip covered by policy. Table of Benefits shows 100% of sum insured payable for Accident Death. Nominee must submit death certificate, police report, detailed sequence of events, post mortem report. Decision: APPROVED - Full sum insured to nominee.",

    # 9
    "Only persons named as Insured in Policy Schedule are covered (General Condition - Insured definition). Policy does not provide coverage for non-insured persons. Mother must be separately insured under group policy or have own policy. Decision: REJECTED - No coverage for non-insured persons.",

    # 10
    "Diabetes and hypertension are pre-existing diseases (diagnosed/treated within 48 months before policy). General Exclusion 2 excludes pre-existing conditions. Base cover only covers accidents. For illness, need Endorsement no.1 (Emergency Medical Expenses-Illness), which also excludes pre-existing. Only Endorsement no.3 - Pre-existing Condition in Life Threatening Situation covers ICU admission for pre-existing if life-threatening. Decision: REJECTED unless Endorsement no.3 opted and condition truly life-threatening."
]


async def evaluate_rag_system():
    """Run evaluation on the RAG system using RAGAS metrics."""
    with open("rag_evaluation_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)

    
    # # Process all queries through the RAG pipeline
    # for i, query in enumerate(queries):
    #     print(f"\nProcessing query {i+1}/{len(queries)}...")
    #     result = await run_rag_pipeline(query)
        
    #     if result:
    #         results.append({
    #             "question": query,
    #             "contexts": result["retrieved_texts"],
    #             "answer": result["answer"],
    #             "ground_truth": ground_truths[i]
    #         })
    #     else:
    #         print(f"Failed to process query {i+1}")

    # Convert results to Dataset format for RAGAS

    
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "answer": [r["answer"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results]
    })
    
    print("\n🚀 Running RAGAS Evaluation...\n")
    # Calculate RAGAS metrics with configured LLM and embeddings
    scores = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            answer_correctness
        ],
        embeddings=embedding_model,
        # llm=llm  # Optional: comment out if you don't want to use Gemini
    )
    
    print("\nEvaluation Results:")
    for metric_name, score in scores.items():
        print(f"{metric_name}: {score:.4f}")
    
    # Save results to file
    with open("rag_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to rag_evaluation_results.json")

# async def run_rag_pipeline(query: str) -> Dict[str, Any]:
#     """Run the complete RAG pipeline on a single query."""
#     try:
#         # Step 1: Rewrite the query
#         rewritten_query = await query_rewriter.rewrite_query(query)
#         if not rewritten_query:
#             raise Exception("Query rewriting failed")

#         # Step 2: Retrieve chunks
#         chunks = retriever.retrive_Chunks(rewritten_query)
#         if not chunks:
#             raise Exception("No relevant chunks found")

#         # Step 3: Rerank chunks
#         reranked_chunks = await reranker.rerank_chunks(rewritten_query, chunks)
#         if not reranked_chunks:
#             raise Exception("Chunk reranking failed")

#         # Step 4: Generate answer
#         answer_response = await answer_generator.generate_answer(rewritten_query, reranked_chunks)
        
#         # Extract text from chunks for evaluation
#         # Chunks are dictionaries with text content
#         retrieved_texts = [chunk['text'] if isinstance(chunk, dict) else chunk for chunk in reranked_chunks]
        
#         return {
#             "retrieved_texts": retrieved_texts,
#             "answer": answer_response["answer"]
#         }
#     except Exception as e:
#         print(f"Error processing query: {str(e)}")
#         return None



if __name__ == "__main__":
    asyncio.run(evaluate_rag_system())
