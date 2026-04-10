"""
Parallel single-turn RAGAS evaluation pipeline using OpenAI models.

This keeps the same input schema as `ragas/openai_eval`, but evaluates up to
5 questions concurrently. Metrics inside each question still run sequentially.

Input JSON records are expected in this shape:
{
    "question_id": "Q_001",
    "user_input": "...",
    "retrieved_contexts": ["..."],
    "reference_contexts": ["..."],
    "retrieved_context_ids": ["..."],
    "reference_context_ids": ["..."],
    "response": "...",
    "reference": "..."
}

Outputs:
    - ragas_single_turn_parallel_results.csv
    - ragas_single_turn_parallel_aggregated_results.txt
"""

from __future__ import annotations

import asyncio
import inspect
import json
import numbers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_JSON = "combined_evaluation_dataset_semantic_chatgpt.json"
OUTPUT_CSV = "ragas_single_turn_parallel_results_semantic_chatgpt.csv"
OUTPUT_TXT = "ragas_single_turn_parallel_aggregated_results_semantic_chatgpt.txt"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_PARALLEL_QUESTIONS = 5

REQUIRED_FIELDS = {
    "question_id",
    "user_input",
    "retrieved_contexts",
    "reference_contexts",
    "retrieved_context_ids",
    "reference_context_ids",
    "response",
    "reference",
}

METRIC_LABELS = {
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "response_relevancy": "Response Relevancy",
    "faithfulness": "Faithfulness",
    "answer_accuracy": "Answer Accuracy",
    "context_relevance": "Context Relevance",
    "response_groundedness": "Response Groundedness",
    "semantic_similarity": "Semantic Similarity",
    "bleu_score": "BLEU Score",
    "rouge1": "ROUGE-1",
    "rougeL": "ROUGE-L",
}


def prefer_installed_ragas() -> None:
    """
    Prevent the repo's local `ragas/` folder from shadowing the installed library.
    """
    project_root = PROJECT_ROOT.resolve()
    cleaned_path: list[str] = []
    for entry in sys.path:
        try:
            resolved = Path(entry or os.getcwd()).resolve()
        except OSError:
            cleaned_path.append(entry)
            continue
        if resolved == project_root:
            continue
        cleaned_path.append(entry)
    sys.path[:] = cleaned_path


prefer_installed_ragas()

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'langchain-openai'. Install the evaluator dependencies "
        "with: pip install -r requirements.txt"
    ) from exc

try:
    try:
        from ragas.dataset_schema import SingleTurnSample
    except ImportError:
        from ragas import SingleTurnSample

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        from ragas.embeddings.base import LangchainEmbeddingsWrapper

    try:
        from ragas.llms import LangchainLLMWrapper
    except ImportError:
        from ragas.llms.base import LangchainLLMWrapper

    import ragas.metrics as ragas_metrics
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'ragas'. Install the evaluator dependencies with: "
        "pip install -r requirements.txt"
    ) from exc


def load_records(json_path: Path) -> list[dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as file:
        raw_data = json.load(file)

    if isinstance(raw_data, list):
        records = raw_data
    elif isinstance(raw_data, dict):
        if REQUIRED_FIELDS.issubset(raw_data.keys()):
            records = [raw_data]
        else:
            records = None
            for key in ("questions", "data", "records", "items", "results"):
                value = raw_data.get(key)
                if isinstance(value, list):
                    print(f"Detected wrapper key '{key}' in the JSON payload.")
                    records = value
                    break
            if records is None:
                raise ValueError(
                    "Input JSON must be either a list of records, a single record, "
                    "or a dict containing a list under one of: "
                    "'questions', 'data', 'records', 'items', 'results'."
                )
    else:
        raise TypeError("Input JSON must deserialize to a list or dict.")

    if not records:
        raise ValueError("Input dataset is empty.")

    validated_records: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise TypeError(f"Record #{index} is not a JSON object.")
        missing = sorted(REQUIRED_FIELDS - set(record.keys()))
        if missing:
            raise KeyError(f"Record #{index} is missing required fields: {missing}")
        validated_records.append(record)

    return validated_records


def to_string_list(value: Any, field_name: str, question_id: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field_name} for {question_id} must be a list.")
    return [str(item) for item in value if item is not None]


def to_id_list(value: Any, field_name: str, question_id: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field_name} for {question_id} must be a list.")
    return [str(item) for item in value if item is not None]


def record_to_sample(record: dict[str, Any]) -> SingleTurnSample:
    question_id = str(record["question_id"])
    return SingleTurnSample(
        user_input=str(record["user_input"]),
        retrieved_contexts=to_string_list(
            record["retrieved_contexts"], "retrieved_contexts", question_id
        ),
        reference_contexts=to_string_list(
            record["reference_contexts"], "reference_contexts", question_id
        ),
        retrieved_context_ids=to_id_list(
            record["retrieved_context_ids"], "retrieved_context_ids", question_id
        ),
        reference_context_ids=to_id_list(
            record["reference_context_ids"], "reference_context_ids", question_id
        ),
        response=str(record["response"]),
        reference=str(record["reference"]),
    )


def record_to_ragas_payload(record: dict[str, Any]) -> dict[str, Any]:
    sample = record_to_sample(record)
    return sample_to_kwargs(sample)


def resolve_metric_class(label: str, *candidate_names: str) -> type:
    for name in candidate_names:
        metric_cls = getattr(ragas_metrics, name, None)
        if metric_cls is not None:
            return metric_cls
    raise ImportError(
        f"Unable to find a RAGAS metric for '{label}'. "
        f"Tried: {', '.join(candidate_names)}"
    )


def build_metric(
    metric_cls: type,
    *,
    label: str | None = None,
    llm: Any = None,
    embeddings: Any = None,
    **overrides: Any,
) -> Any:
    try:
        signature = inspect.signature(metric_cls)
        if llm is not None and "llm" in signature.parameters:
            overrides["llm"] = llm
        if embeddings is not None and "embeddings" in signature.parameters:
            overrides["embeddings"] = embeddings
    except (TypeError, ValueError):
        pass
    try:
        return metric_cls(**overrides)
    except ImportError as exc:
        metric_label = label or metric_cls.__name__
        raise ImportError(
            f"Unable to initialize the '{metric_label}' metric. "
            "Install the evaluator dependencies with: pip install -r requirements.txt"
        ) from exc


def get_metric_specs(ragas_llm: Any, ragas_embeddings: Any) -> dict[str, Any]:
    rouge_cls = resolve_metric_class("Rouge Score", "RougeScore")

    return {
        "context_precision": build_metric(
            resolve_metric_class(
                "Context Precision",
                "ContextPrecision",
                "LLMContextPrecisionWithReference",
            ),
            label="Context Precision",
            llm=ragas_llm,
        ),
        "context_recall": build_metric(
            resolve_metric_class(
                "Context Recall",
                "ContextRecall",
                "LLMContextRecall",
            ),
            label="Context Recall",
            llm=ragas_llm,
        ),
        "response_relevancy": build_metric(
            resolve_metric_class(
                "Response Relevancy",
                "ResponseRelevancy",
                "AnswerRelevancy",
            ),
            label="Response Relevancy",
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        ),
        "faithfulness": build_metric(
            resolve_metric_class("Faithfulness", "Faithfulness"),
            label="Faithfulness",
            llm=ragas_llm,
        ),
        "answer_accuracy": build_metric(
            resolve_metric_class("Answer Accuracy", "AnswerAccuracy"),
            label="Answer Accuracy",
            llm=ragas_llm,
        ),
        "context_relevance": build_metric(
            resolve_metric_class("Context Relevance", "ContextRelevance"),
            label="Context Relevance",
            llm=ragas_llm,
        ),
        "response_groundedness": build_metric(
            resolve_metric_class("Response Groundedness", "ResponseGroundedness"),
            label="Response Groundedness",
            llm=ragas_llm,
        ),
        "semantic_similarity": build_metric(
            resolve_metric_class("Semantic Similarity", "SemanticSimilarity"),
            label="Semantic Similarity",
            embeddings=ragas_embeddings,
        ),
        "bleu_score": build_metric(
            resolve_metric_class("BLEU Score", "BleuScore"),
            label="BLEU Score",
        ),
        "rouge1": build_metric(
            rouge_cls,
            label="ROUGE-1",
            rouge_type="rouge1",
            mode="fmeasure",
        ),
        "rougeL": build_metric(
            rouge_cls,
            label="ROUGE-L",
            rouge_type="rougeL",
            mode="fmeasure",
        ),
    }


def sample_to_kwargs(sample: SingleTurnSample) -> dict[str, Any]:
    if hasattr(sample, "to_dict"):
        return sample.to_dict()
    return {
        "user_input": sample.user_input,
        "retrieved_contexts": sample.retrieved_contexts,
        "reference_contexts": sample.reference_contexts,
        "retrieved_context_ids": sample.retrieved_context_ids,
        "reference_context_ids": sample.reference_context_ids,
        "response": sample.response,
        "reference": sample.reference,
    }


def extract_score_value(result: Any) -> float | None:
    if result is None:
        return None
    if isinstance(result, numbers.Real) and not isinstance(result, bool):
        return float(result)
    for attr in ("value", "score"):
        value = getattr(result, attr, None)
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            return float(value)
    return None


async def score_metric_async(metric: Any, sample: SingleTurnSample) -> float | None:
    payload = sample_to_kwargs(sample)

    if hasattr(metric, "single_turn_ascore"):
        return extract_score_value(await metric.single_turn_ascore(sample))

    if hasattr(metric, "ascore"):
        try:
            return extract_score_value(await metric.ascore(**payload))
        except TypeError:
            return extract_score_value(await metric.ascore(sample))

    if hasattr(metric, "single_turn_score"):
        return extract_score_value(await asyncio.to_thread(metric.single_turn_score, sample))

    if hasattr(metric, "score"):
        try:
            return extract_score_value(await asyncio.to_thread(metric.score, **payload))
        except TypeError:
            return extract_score_value(await asyncio.to_thread(metric.score, sample))

    raise AttributeError(
        f"Metric '{metric.__class__.__name__}' does not expose a supported scoring method."
    )


def build_clients() -> tuple[Any, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY_GEN")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY_GEN was not found in .env or the environment.")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)


def mean_or_none(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def serialize_list_field(value: list[str]) -> str:
    return json.dumps(value, ensure_ascii=True)


def compute_overall_average(results_df: pd.DataFrame, metric_columns: list[str]) -> float | None:
    averages = [
        mean_or_none(results_df[column])
        for column in metric_columns
        if column in results_df.columns
    ]
    valid_averages = [value for value in averages if value is not None]
    if not valid_averages:
        return None
    return float(sum(valid_averages) / len(valid_averages))


def write_average_report(
    output_path: Path,
    input_path: Path,
    results_df: pd.DataFrame,
    metric_columns: list[str],
) -> None:
    rows_with_errors = int(results_df["errors"].astype(str).str.len().gt(0).sum())
    lines = [
        "RAGAS Single-Turn Parallel Evaluation Summary",
        "=" * 49,
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Input file: {input_path}",
        f"Rows evaluated: {len(results_df)}",
        f"Rows with metric errors: {rows_with_errors}",
        f"Parallel question workers: {MAX_PARALLEL_QUESTIONS}",
        "",
        "Average scores:",
    ]

    for metric in metric_columns:
        avg = mean_or_none(results_df[metric])
        valid_rows = int(results_df[metric].notna().sum())
        label = METRIC_LABELS.get(metric, metric)
        if avg is None:
            lines.append(f"- {label}: N/A (valid rows: {valid_rows}/{len(results_df)})")
        else:
            lines.append(
                f"- {label}: {avg:.6f} (valid rows: {valid_rows}/{len(results_df)})"
            )

    overall_average = compute_overall_average(results_df, metric_columns)
    lines.extend(
        [
            "",
            "Overall summary:",
            (
                f"- Overall average across computed metrics: {overall_average:.6f}"
                if overall_average is not None
                else "- Overall average across computed metrics: N/A"
            ),
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def score_record(
    record_index: int,
    total_records: int,
    record: dict[str, Any],
    metric_map: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        question_id = str(record["question_id"])
        print(f"Starting {record_index}/{total_records}: {question_id}")

        ragas_payload = record_to_ragas_payload(record)
        sample = record_to_sample(record)
        row = {
            "_row_index": record_index,
            "question_id": question_id,
            "user_input": str(record["user_input"]),
            "response": str(record["response"]),
            "reference": str(record["reference"]),
            "retrieved_contexts": serialize_list_field(
                ragas_payload.get("retrieved_contexts", [])
            ),
            "reference_contexts": serialize_list_field(
                ragas_payload.get("reference_contexts", [])
            ),
            "retrieved_contexts_count": len(sample.retrieved_contexts or []),
            "reference_contexts_count": len(sample.reference_contexts or []),
            "retrieved_context_ids": serialize_list_field(
                [str(item) for item in (sample.retrieved_context_ids or [])]
            ),
            "reference_context_ids": serialize_list_field(
                [str(item) for item in (sample.reference_context_ids or [])]
            ),
            "errors": "",
        }

        row_errors: dict[str, str] = {}
        for metric_name, metric in metric_map.items():
            try:
                row[metric_name] = await score_metric_async(metric, sample)
            except Exception as exc:
                row[metric_name] = None
                row_errors[metric_name] = str(exc)

        if row_errors:
            row["errors"] = json.dumps(row_errors, ensure_ascii=True)

        print(f"Finished {record_index}/{total_records}: {question_id}")
        return row


async def main_async() -> None:
    input_path = SCRIPT_DIR / INPUT_JSON
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    print(f"Loading dataset from {input_path}")
    records = load_records(input_path)
    print(f"Loaded {len(records)} record(s)")
    print(f"Running with up to {MAX_PARALLEL_QUESTIONS} questions in parallel")

    ragas_llm, ragas_embeddings = build_clients()
    metric_map = get_metric_specs(ragas_llm, ragas_embeddings)
    metric_columns = list(metric_map.keys())
    semaphore = asyncio.Semaphore(MAX_PARALLEL_QUESTIONS)

    tasks = [
        score_record(index, len(records), record, metric_map, semaphore)
        for index, record in enumerate(records, start=1)
    ]
    rows = await asyncio.gather(*tasks)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("_row_index").drop(columns=["_row_index"])

    output_csv_path = SCRIPT_DIR / OUTPUT_CSV
    results_df.to_csv(output_csv_path, index=False)

    output_txt_path = SCRIPT_DIR / OUTPUT_TXT
    write_average_report(output_txt_path, input_path, results_df, metric_columns)

    print("\nPer-row results saved to:")
    print(output_csv_path)
    print("\nAverage score summary saved to:")
    print(output_txt_path)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
