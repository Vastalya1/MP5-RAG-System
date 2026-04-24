"""
LangGraph-based query decomposition for complex RAG questions.
"""

from __future__ import annotations

import contextvars
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from shared.logging_utils import get_logger, log_error, log_info, log_query_error, log_query_step

sys.path.append(str(Path(__file__).resolve().parent.parent))

from orchestration.rag_pipeline_Chatgpt import (
    RAGSubQueryProcessor,
    aggregate_retrieval_debug,
    merge_sources,
)


logger = get_logger(__name__)


class DecompositionState(TypedDict):
    query: str
    scope: str
    username: Optional[str]
    collection_name: Optional[str]
    document_filter: Optional[str]
    should_decompose: bool
    sub_queries: List[str]
    sub_query_results: List[Dict[str, Any]]
    answer: Optional[str]
    justification: Optional[str]
    sources: List[Dict[str, Any]]
    rewritten_query: Optional[str]
    success: bool
    needs_web_scraping: bool
    error: Optional[str]
    metadata: Dict[str, Any]


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output")
    return json.loads(cleaned[start : end + 1])


class QueryDecomposer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.prompt = """You are a query decomposition planner for a medical insurance RAG assistant.

Decide whether the user's question should be split into smaller independent sub-questions before retrieval.

Only decompose when the query asks about multiple distinct aspects that can be answered separately and then combined.
Examples of decomposable patterns:
- coverage plus waiting period
- eligibility plus required documents
- benefits plus exclusions plus claim steps

Rules:
- Return at most 3 sub-queries.
- Each sub-query must be standalone and explicit.
- Preserve the user's original intent.
- Avoid overlap and redundancy across sub-queries.
- If the question is already focused enough, do not decompose it.

Return JSON only in this exact shape:
{"should_decompose": true, "sub_queries": ["...", "..."], "reason": "..."}"""

    def plan(self, query: str) -> Dict[str, Any]:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=250,
            )
            content = (response.choices[0].message.content or "").strip() if response and response.choices else ""
            plan = _extract_json_object(content)
            sub_queries = [
                str(item).strip()
                for item in plan.get("sub_queries", []) or []
                if str(item).strip()
            ]
            deduped_sub_queries: List[str] = []
            seen: set[str] = set()
            for sub_query in sub_queries:
                key = sub_query.lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped_sub_queries.append(sub_query)

            should_decompose = bool(plan.get("should_decompose")) and len(deduped_sub_queries) > 1
            return {
                "should_decompose": should_decompose,
                "sub_queries": deduped_sub_queries[:3] if should_decompose else [query],
                "reason": str(plan.get("reason", "")).strip(),
            }
        except Exception as e:
            log_error(logger, "query_decomposition_planning_failed", error_type=type(e).__name__, error=str(e))
            log_query_error(
                logger,
                "query_decomposition_plan",
                generated=[query],
                error_type=type(e).__name__,
                error=str(e),
            )
            return {
                "should_decompose": False,
                "sub_queries": [query],
                "reason": "Decomposition planning failed; continuing with original query.",
            }


class SubQuerySynthesizer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.prompt = """You are synthesizing a final answer for a medical insurance RAG assistant.

Original user query:
{query}

Sub-query results:
{sub_query_results}

Instructions:
- Combine the sub-query answers into one coherent final answer.
- Directly answer the original user query.
- If some parts are unclear or missing, say so instead of guessing.
- Use plain text only.
- Keep the final answer grounded in the provided sub-query results.

Respond in this format only:
Answer:
[final answer]

Justification:
[brief synthesis justification with referenced sections if available]"""

    def synthesize(self, query: str, sub_query_results: List[Dict[str, Any]]) -> Dict[str, str]:
        try:
            results_text_parts: List[str] = []
            for index, result in enumerate(sub_query_results, start=1):
                results_text_parts.append(
                    "\n".join(
                        [
                            f"Sub-query {index}: {result.get('sub_query', '')}",
                            f"Rewritten: {result.get('rewritten_query', '')}",
                            f"Answer: {result.get('answer', '')}",
                            f"Justification: {result.get('justification', '')}",
                        ]
                    )
                )

            prompt = self.prompt.format(
                query=query,
                sub_query_results="\n\n".join(results_text_parts),
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You synthesize grounded insurance-policy answers from sub-query results.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=500,
            )
            content = (response.choices[0].message.content or "").strip() if response and response.choices else ""
            answer_parts = content.split("Justification:")
            answer = answer_parts[0].replace("Answer:", "").strip()
            justification = answer_parts[1].strip() if len(answer_parts) > 1 else ""
            return {"answer": answer, "justification": justification}
        except Exception as e:
            log_error(logger, "subquery_synthesis_failed", error_type=type(e).__name__, error=str(e))
            fallback_answer = " ".join(
                result.get("answer", "") for result in sub_query_results if result.get("answer")
            ).strip()
            fallback_justification = " ".join(
                result.get("justification", "")
                for result in sub_query_results
                if result.get("justification")
            ).strip()
            log_query_error(
                logger,
                "sub_query_synthesis",
                generated=fallback_answer,
                error_type=type(e).__name__,
                error=str(e),
            )
            return {
                "answer": fallback_answer or "I could not synthesize a final answer from the sub-query results.",
                "justification": fallback_justification,
            }


class QueryDecompositionOrchestrator:
    def __init__(self, api_key: str, rag_processor: RAGSubQueryProcessor):
        self.decomposer = QueryDecomposer(api_key)
        self.synthesizer = SubQuerySynthesizer(api_key)
        self.rag_processor = rag_processor
        self.graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(DecompositionState)
        builder.add_node("plan_decomposition", self._plan_decomposition_node)
        builder.add_node("process_single_query", self._process_single_query_node)
        builder.add_node("process_sub_queries", self._process_sub_queries_node)
        builder.add_node("synthesize_answer", self._synthesize_answer_node)

        builder.add_edge(START, "plan_decomposition")
        builder.add_conditional_edges(
            "plan_decomposition",
            self._route_after_planning,
            {"single": "process_single_query", "decompose": "process_sub_queries"},
        )
        builder.add_edge("process_single_query", END)
        builder.add_edge("process_sub_queries", "synthesize_answer")
        builder.add_edge("synthesize_answer", END)
        return builder

    def _plan_decomposition_node(self, state: DecompositionState) -> Dict[str, Any]:
        plan = self.decomposer.plan(state["query"])
        log_info(
            logger,
            "query_decomposition_plan_created",
            should_decompose=plan["should_decompose"],
            sub_query_count=len(plan["sub_queries"]),
        )
        log_query_step(
            logger,
            "query_decomposition_plan",
            generated=plan["sub_queries"],
            should_decompose=plan["should_decompose"],
            reason=plan.get("reason"),
        )
        return {
            "should_decompose": plan["should_decompose"],
            "sub_queries": plan["sub_queries"],
            "metadata": {
                **state.get("metadata", {}),
                "decomposition_reason": plan.get("reason"),
                "planned_sub_queries": plan["sub_queries"],
            },
        }

    def _route_after_planning(self, state: DecompositionState) -> Literal["single", "decompose"]:
        return "decompose" if state.get("should_decompose") else "single"

    def _process_single_query_node(self, state: DecompositionState) -> Dict[str, Any]:
        result = self.rag_processor.process_sync(
            query=state["query"],
            scope=state["scope"],
            username=state.get("username"),
            collection_name=state.get("collection_name"),
            document_filter=state.get("document_filter"),
        )
        return {
            "sub_query_results": [{**result, "sub_query": state["query"]}],
            "answer": result.get("answer"),
            "justification": result.get("justification"),
            "sources": result.get("sources", []),
            "rewritten_query": result.get("rewritten_query"),
            "success": result.get("success", False),
            "needs_web_scraping": result.get("needs_web_scraping", False),
            "error": result.get("error"),
            "metadata": {
                **state.get("metadata", {}),
                "decomposition_used": False,
                "sub_query_count": 1,
                "sub_query_results": [
                    {
                        "sub_query": state["query"],
                        "rewritten_query": result.get("rewritten_query"),
                    }
                ],
            },
        }

    def _process_sub_queries_node(self, state: DecompositionState) -> Dict[str, Any]:
        sub_queries = (state.get("sub_queries") or [])[:3]
        results: List[Dict[str, Any]] = []

        max_workers = max(1, min(3, len(sub_queries)))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="query-decomp") as executor:
            future_map = {
                executor.submit(
                    contextvars.copy_context().run,
                    self.rag_processor.process_sync,
                    sub_query,
                    state["scope"],
                    state.get("username"),
                    state.get("collection_name"),
                    state.get("document_filter"),
                ): sub_query
                for sub_query in sub_queries
            }

            for future in as_completed(future_map):
                sub_query = future_map[future]
                try:
                    result = future.result()
                except Exception as e:
                    log_query_error(
                        logger,
                        "sub_query_processing",
                        generated=str(e),
                        sub_query=sub_query,
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                    result = {
                        "answer": f"An error occurred while processing this sub-query: {e}",
                        "justification": None,
                        "sources": [],
                        "rewritten_query": sub_query,
                        "success": False,
                        "needs_web_scraping": False,
                        "error": str(e),
                        "retrieval_debug": {"source_counts": {}, "top_chunks": [], "top_chunk_count": 0},
                        "top_chunk_distance": 2.0,
                    }

                results.append({**result, "sub_query": sub_query})
                log_query_step(
                    logger,
                    "sub_query_processed",
                    generated={
                        "sub_query": sub_query,
                        "rewritten_query": result.get("rewritten_query"),
                        "answer": result.get("answer", ""),
                    },
                    success=result.get("success", False),
                    needs_web_scraping=result.get("needs_web_scraping", False),
                )

        results.sort(key=lambda item: sub_queries.index(item.get("sub_query", "")))
        aggregated_sources = merge_sources([result.get("sources", []) for result in results])
        overall_needs_web_scraping = bool(results) and all(
            result.get("needs_web_scraping", False) for result in results
        )

        return {
            "sub_query_results": results,
            "sources": aggregated_sources,
            "success": any(result.get("success", False) for result in results),
            "needs_web_scraping": overall_needs_web_scraping,
            "metadata": {
                **state.get("metadata", {}),
                "decomposition_used": True,
                "sub_query_count": len(results),
            },
        }

    def _synthesize_answer_node(self, state: DecompositionState) -> Dict[str, Any]:
        sub_query_results = state.get("sub_query_results", [])
        synthesis = self.synthesizer.synthesize(state["query"], sub_query_results)
        log_query_step(
            logger,
            "sub_query_synthesis",
            generated=synthesis.get("answer"),
            sub_query_count=len(sub_query_results),
            justification=synthesis.get("justification"),
        )
        rewritten_queries = [
            result.get("rewritten_query", "")
            for result in sub_query_results
            if result.get("rewritten_query")
        ]
        top_distances = [result.get("top_chunk_distance", 2.0) for result in sub_query_results]

        return {
            "answer": synthesis.get("answer"),
            "justification": synthesis.get("justification"),
            "rewritten_query": " | ".join(rewritten_queries),
            "success": any(result.get("success", False) for result in sub_query_results),
            "error": None,
            "metadata": {
                **state.get("metadata", {}),
                "sub_query_results": [
                    {
                        "sub_query": result.get("sub_query"),
                        "rewritten_query": result.get("rewritten_query"),
                        "success": result.get("success", False),
                    }
                    for result in sub_query_results
                ],
                "top_chunk_distances": top_distances,
            },
            "sources": merge_sources([result.get("sources", []) for result in sub_query_results]),
            "needs_web_scraping": state.get("needs_web_scraping", False),
        }

    async def process_query(
        self,
        query: str,
        scope: str = "shared",
        username: Optional[str] = None,
        collection_name: Optional[str] = None,
        document_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        initial_state: DecompositionState = {
            "query": query,
            "scope": scope,
            "username": username,
            "collection_name": collection_name,
            "document_filter": document_filter,
            "should_decompose": False,
            "sub_queries": [],
            "sub_query_results": [],
            "answer": None,
            "justification": None,
            "sources": [],
            "rewritten_query": None,
            "success": False,
            "needs_web_scraping": False,
            "error": None,
            "metadata": {},
        }

        result = await self.graph.ainvoke(initial_state)
        sub_query_results = result.get("sub_query_results", [])
        retrieval_debug = (
            sub_query_results[0].get("retrieval_debug")
            if len(sub_query_results) == 1
            else aggregate_retrieval_debug(sub_query_results)
        )

        return {
            "answer": result.get("answer"),
            "justification": result.get("justification"),
            "sources": result.get("sources", []),
            "rewritten_query": result.get("rewritten_query"),
            "success": result.get("success", False),
            "needs_web_scraping": result.get("needs_web_scraping", False),
            "error": result.get("error"),
            "retrieval_debug": retrieval_debug,
            "sub_query_results": sub_query_results,
            "metadata": result.get("metadata", {}),
        }
