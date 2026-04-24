import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))

from orchestration.orchestrator_Chatgpt import create_orchestrator
from output.answerGeneration_Chatgpt import AnswerGenerator
from queryRewriter.rewriting_Chatgpt import QueryRewriter
from retriever.reranking_Chatgpt import ChunkReranker
from retriever.retrival import retrivalModel
from shared.chroma_config import get_personal_collection_name, get_shared_collection_name
from shared.logging_utils import (
    configure_logging,
    current_query_log_path,
    get_logger,
    log_query_error,
    new_transaction_id,
    reset_query_id,
    reset_transaction_id,
    set_transaction_id,
    start_query_log,
)


logger = get_logger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _query_phase_output_dir() -> Path:
    path = _repo_root() / "Logs" / "query_phases"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_text_block(lines: list[str]) -> str:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _parse_query_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []

    raw_text = log_path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in raw_text.split("=" * 100) if block.strip()]
    entries: list[dict[str, Any]] = []

    for block in blocks:
        lines = block.splitlines()
        entry: dict[str, Any] = {"details": {}, "generated": ""}
        section: str | None = None
        active_key: str | None = None
        active_buffer: list[str] = []
        generated_lines: list[str] = []

        def flush_active_buffer() -> None:
            nonlocal active_key, active_buffer
            if active_key is not None:
                entry["details"][active_key] = _normalize_text_block(active_buffer)
                active_key = None
                active_buffer = []

        for line in lines:
            if line == "details:":
                flush_active_buffer()
                section = "details"
                continue
            if line == "generated:":
                flush_active_buffer()
                section = "generated"
                continue

            if section == "generated":
                generated_lines.append(line)
                continue

            if section == "details" and line.startswith("  "):
                active_buffer.append(line[2:])
                continue

            flush_active_buffer()
            if ": " in line:
                key, value = line.split(": ", 1)
                if section == "details":
                    entry["details"][key] = value
                else:
                    entry[key] = value
            elif line.endswith(":") and section == "details":
                active_key = line[:-1]
                active_buffer = []

        flush_active_buffer()
        entry["generated"] = _normalize_text_block(generated_lines)
        entries.append(entry)

    return entries


def _render_lifecycle_report(
    *,
    query: str,
    scope: str,
    username: str | None,
    collection_name: str | None,
    document_filter: str | None,
    query_id: str,
    query_log_path: Path | None,
    result: dict[str, Any],
    phase_entries: list[dict[str, Any]],
) -> str:
    lines = [
        "QUERY LIFECYCLE REPORT",
        "=" * 80,
        f"query_id: {query_id}",
        f"scope: {scope}",
        f"username: {username or '-'}",
        f"collection_name: {collection_name or '-'}",
        f"document_filter: {document_filter or '-'}",
        "",
        "ORIGINAL QUESTION",
        "-" * 80,
        query.strip(),
        "",
        "FINAL RESULT",
        "-" * 80,
        f"route_taken: {result.get('route_taken', '-')}",
        f"success: {result.get('success', False)}",
        f"needs_web_scraping: {result.get('needs_web_scraping', False)}",
        "",
        "response:",
        str(result.get("response", "") or "").strip(),
        "",
    ]

    justification = str(result.get("justification", "") or "").strip()
    if justification:
        lines.extend(["justification:", justification, ""])

    if query_log_path is not None:
        lines.extend(["raw_query_log:", str(query_log_path), ""])

    lines.extend(["QUERY PHASES", "-" * 80])

    if not phase_entries:
        lines.append("No query phase entries were found.")
        return "\n".join(lines).strip() + "\n"

    for index, entry in enumerate(phase_entries, start=1):
        lines.append(f"{index}. {entry.get('step', 'unknown_step')}")
        timestamp = entry.get("time")
        level = entry.get("level")
        logger_name = entry.get("logger")
        if timestamp:
            lines.append(f"   time: {timestamp}")
        if level:
            lines.append(f"   level: {level}")
        if logger_name:
            lines.append(f"   logger: {logger_name}")

        details = entry.get("details", {}) or {}
        if details:
            lines.append("   details:")
            for key, value in details.items():
                value_text = str(value or "").strip()
                if "\n" in value_text:
                    lines.append(f"   - {key}:")
                    for value_line in value_text.splitlines():
                        lines.append(f"     {value_line}")
                else:
                    lines.append(f"   - {key}: {value_text}")

        generated = str(entry.get("generated", "") or "").strip()
        if generated:
            lines.append("   generated:")
            for generated_line in generated.splitlines():
                lines.append(f"     {generated_line}")

        lines.append("")

    return "\n".join(lines).strip() + "\n"


async def _run_query_pipeline_async(
    *,
    query: str,
    scope: str,
    username: str | None,
    collection_name: str | None,
    document_filter: str | None,
) -> dict[str, Any]:
    env_path = _repo_root() / ".env"
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the query pipeline.")

    rewriter = QueryRewriter(api_key)
    retriever = retrivalModel()
    reranker = ChunkReranker(api_key)
    answer_generator = AnswerGenerator(api_key)
    orchestrator = create_orchestrator(
        api_key=api_key,
        rewriter=rewriter,
        retriever=retriever,
        reranker=reranker,
        answer_generator=answer_generator,
    )

    return await orchestrator.process_query(
        query=query,
        scope=scope,
        username=username,
        collection_name=collection_name,
        document_filter=document_filter,
    )


def run_query_pipeline(
    query: str,
    *,
    scope: str = "shared",
    username: str | None = None,
    collection_name: str | None = None,
    document_filter: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    configure_logging()
    query_id = new_transaction_id("phase")
    transaction_token = set_transaction_id(query_id)
    query_token = None
    log_path: Path | None = None

    try:
        resolved_collection_name = collection_name
        if resolved_collection_name is None:
            if scope == "shared":
                resolved_collection_name = get_shared_collection_name()
            elif scope == "personal" and username:
                resolved_collection_name = get_personal_collection_name(username)

        query_token = start_query_log(
            query_id,
            logger=logger,
            username=username,
            scope=scope,
            collection_name=resolved_collection_name,
            selected_document=document_filter,
            generated=query,
        )
        log_path = current_query_log_path()

        try:
            result = asyncio.run(
                _run_query_pipeline_async(
                    query=query,
                    scope=scope,
                    username=username,
                    collection_name=resolved_collection_name,
                    document_filter=document_filter,
                )
            )
        except Exception as exc:
            log_query_error(
                logger,
                "query_pipeline_runner_failed",
                generated=str(exc),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            result = {
                "response": f"Pipeline execution failed: {exc}",
                "justification": None,
                "sources": [],
                "route_taken": "runner_error",
                "success": False,
                "needs_web_scraping": False,
                "error": str(exc),
            }

        phase_entries = _parse_query_log(log_path) if log_path is not None else []
        lifecycle_text = _render_lifecycle_report(
            query=query,
            scope=scope,
            username=username,
            collection_name=resolved_collection_name,
            document_filter=document_filter,
            query_id=query_id,
            query_log_path=log_path,
            result=result,
            phase_entries=phase_entries,
        )

        lifecycle_path = Path(output_path).resolve() if output_path else _query_phase_output_dir() / f"{query_id}_lifecycle.txt"
        lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
        lifecycle_path.write_text(lifecycle_text, encoding="utf-8")

        return {
            "query_id": query_id,
            "query_log_path": str(log_path) if log_path is not None else None,
            "lifecycle_path": str(lifecycle_path),
            "result": result,
        }
    finally:
        if query_token is not None:
            reset_query_id(query_token)
        reset_transaction_id(transaction_token)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full question pipeline and write a lifecycle text report."
    )
    parser.add_argument("--query", required=True, help="Question to run through the full query pipeline.")
    parser.add_argument(
        "--scope",
        default="shared",
        choices=["shared", "personal", "combined"],
        help="Query scope to use.",
    )
    parser.add_argument("--username", default=None, help="Username for personal or combined scope.")
    parser.add_argument("--collection-name", default=None, help="Optional collection name override.")
    parser.add_argument("--document-filter", default=None, help="Optional document name filter.")
    parser.add_argument("--output", default=None, help="Optional lifecycle report output path.")
    args = parser.parse_args()

    run_result = run_query_pipeline(
        args.query,
        scope=args.scope,
        username=args.username,
        collection_name=args.collection_name,
        document_filter=args.document_filter,
        output_path=args.output,
    )

    result = run_result["result"]
    print(f"Lifecycle report: {run_result['lifecycle_path']}")
    if run_result.get("query_log_path"):
        print(f"Query log: {run_result['query_log_path']}")
    print(f"Route taken: {result.get('route_taken')}")
    print("Response:")
    print(result.get("response", ""))


if __name__ == "__main__":
    main()
