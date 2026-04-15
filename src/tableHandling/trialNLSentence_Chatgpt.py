import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


GenerationResult = Tuple[List[str], str, Optional[str], Optional[str]]


SYSTEM_PROMPT = (
    "You are a data extraction specialist focused on high-accuracy insurance documentation."
)


USER_PROMPT_TEMPLATE = """Role: You are a data extraction specialist focused on high-accuracy insurance documentation.

Task: Convert the provided Markdown table into a list of natural language sentences.

Strict Constraints:
1. One Row, One Sentence: Each row in the table must correspond to exactly one full sentence.
2. Strictly generate only the sentences and nothing else.
3. Zero Hallucination/Summarization: Do not add information not present in the table.
4. Reference headers naturally. Avoid repeating identical values across headers.
5. Maintain Terminology: Use the exact technical terms for covers and values as written in the markdown.
6. Handling Multiple Values: If a header has multiple values, include all values.
7. Preserve row order from top to bottom.
8. Every sentence must preserve a clear mapping between values and their corresponding plans. Do not merge or generalize values unless all columns have identical values.
9. Do NOT merge rows or infer relationships not directly present.
10. Use heading into context when it is required to disambiguate values or provide necessary context for understanding the row.
Validation Requirements:
- The table has exactly {expected_rows} data rows (excluding header and separator).
- Return exactly {expected_rows} sentences. One row data equals one sentence Only.
- One sentence per line.
- Do not Halucinate or summarize. Use only the information in the table.
- If the table is empty, return an empty list.
- Do Not add reasoning steps, explanations, or any text other than the sentences themselves.
- This is a strict formatting task. Any deviation from the required format is incorrect.
Table Heading: {heading_context}, Markdown Input:
{markdown}

Example Format:
The [COVER NAME] is [VALUE] under [HEADER-1] and [VALUE] under [HEADER-2]."""


def _parse_markdown_row(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator_row(cells: List[str]) -> bool:
    if not cells:
        return False
    return all(re.fullmatch(r"[:\-\s]+", cell or "") is not None for cell in cells)


def parse_markdown_table(markdown: str) -> Tuple[List[str], List[List[str]]]:
    raw_lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    table_lines = [line for line in raw_lines if line.startswith("|") and line.endswith("|")]

    if len(table_lines) < 2:
        return [], []

    header = _parse_markdown_row(table_lines[0])
    idx = 1
    if idx < len(table_lines) and _is_separator_row(_parse_markdown_row(table_lines[idx])):
        idx += 1

    data_rows = [_parse_markdown_row(line) for line in table_lines[idx:]]
    return header, data_rows


def normalize_sentence_lines(text: str) -> List[str]:
    cleaned = text.replace("```", "").strip()
    if not cleaned:
        return []

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    sentences: List[str] = []

    for line in lines:
        line = re.sub(r"^\s*(?:\d+[\).\:-]\s*|[-*]\s+)", "", line).strip()
        if line:
            sentences.append(line)

    if len(sentences) <= 1 and "\n" not in cleaned:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", cleaned)
        fallback = [re.sub(r"^\s*(?:\d+[\).\:-]\s*|[-*]\s+)", "", p).strip() for p in parts if p.strip()]
        if len(fallback) > len(sentences):
            sentences = fallback

    return sentences


def rows_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    if not headers:
        return ""
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = []
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        row_lines.append("| " + " | ".join(padded[: len(headers)]) + " |")
    return "\n".join([header_line, separator_line, *row_lines])


def build_heading_context(heading: Optional[str]) -> str:
    normalized = (heading or "").strip()
    if not normalized:
        return ""
    return f"Table Heading Context:\n{normalized}\n\n"


def describe_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def generate_sentences_for_table(
    client: OpenAI,
    model: str,
    markdown: str,
    expected_rows: int,
    heading: Optional[str] = None,
    max_tokens: int = 12000,
    max_attempts: int = 3,
) -> GenerationResult:
    if not markdown.strip() or expected_rows <= 0:
        return [], "ok", None, None

    heading_context = build_heading_context(heading)
    prompt = USER_PROMPT_TEMPLATE.format(
        expected_rows=expected_rows,
        heading_context=heading_context,
        markdown=markdown,
    )
    last_sentences: List[str] = []
    last_error_type: Optional[str] = None
    last_error_detail: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            last_error_type = "api_exception"
            last_error_detail = f"attempt {attempt}: {describe_exception(exc)}"
            continue

        if not response:
            last_error_type = "empty_response"
            last_error_detail = f"attempt {attempt}: API returned no response object."
            continue

        if not response.choices:
            last_error_type = "empty_choices"
            last_error_detail = f"attempt {attempt}: API response had no choices."
            continue

        content = (response.choices[0].message.content or "").strip()
        sentences = normalize_sentence_lines(content)
        last_sentences = sentences
        if sentences:
            return sentences, "ok", None, None

        last_error_type = "empty_parsed_output"
        last_error_detail = f"attempt {attempt}: model output could not be parsed into sentences."

    if last_sentences:
        return last_sentences, "ok", None, None
    return [], "error", last_error_type or "unknown_generation_failure", last_error_detail


def generate_sentences_chunked(
    client: OpenAI,
    model: str,
    headers: List[str],
    data_rows: List[List[str]],
    heading: Optional[str] = None,
    chunk_size: int = 25,
) -> GenerationResult:
    if not data_rows:
        return [], "ok", None, None
    if not headers:
        return [], "error", "missing_headers", "Chunked generation requires parsed table headers."

    all_sentences: List[str] = []
    for start in range(0, len(data_rows), chunk_size):
        chunk_rows = data_rows[start : start + chunk_size]
        markdown_chunk = rows_to_markdown(headers, chunk_rows)
        chunk_sentences, chunk_status, chunk_error_type, chunk_error_detail = generate_sentences_for_table(
            client=client,
            model=model,
            markdown=markdown_chunk,
            expected_rows=len(chunk_rows),
            heading=heading,
            max_tokens=6000,
            max_attempts=3,
        )

        if chunk_status != "ok":
            chunk_label = f"chunk rows {start + 1}-{start + len(chunk_rows)}"
            detail = chunk_error_detail or f"{chunk_label} failed."
            return all_sentences, "error", chunk_error_type or "chunk_generation_failed", f"{chunk_label}: {detail}"

        all_sentences.extend(chunk_sentences)

    return all_sentences, "ok_chunked", None, None


def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON in {path}, found: {type(payload).__name__}")
    return payload


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_openai_client() -> OpenAI:
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in .env or environment variables.")
    return OpenAI(api_key=api_key)


def enrich_tables_with_sentences(
    tables: List[Dict[str, Any]],
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
    sleep_seconds: float = 0.2,
) -> List[Dict[str, Any]]:
    active_client = client or create_openai_client()
    enriched: List[Dict[str, Any]] = []
    total = len(tables)

    for idx, item in enumerate(tables, start=1):
        markdown = str(item.get("markdown", "") or "")
        heading = str(item.get("heading", "") or "").strip() or None
        headers, data_rows = parse_markdown_table(markdown)
        expected_rows = len(data_rows)
        error_type: Optional[str] = None
        error_detail: Optional[str] = None

        print(f"[{idx}/{total}] page={item.get('page_number')} rows={expected_rows} -> generating...")
        sentences, status, error_type, error_detail = generate_sentences_for_table(
            client=active_client,
            model=model,
            markdown=markdown,
            expected_rows=expected_rows,
            heading=heading,
        )

        if status in {"mismatch", "error"} and expected_rows > 0:
            print(f"[{idx}/{total}] {status} detected -> chunked fallback...")
            chunked_sentences, chunked_status, chunked_error_type, chunked_error_detail = generate_sentences_chunked(
                client=active_client,
                model=model,
                headers=headers,
                data_rows=data_rows,
                heading=heading,
            )
            if chunked_status in {"ok", "ok_chunked"}:
                sentences = chunked_sentences
                status = "ok_chunked"
                error_type = None
                error_detail = None
            else:
                sentences = chunked_sentences
                status = "error"
                error_type = chunked_error_type or "chunked_generation_failed"
                error_detail = chunked_error_detail or "Chunked fallback failed."

        if status == "error" and error_type:
            print(f"[{idx}/{total}] error_type={error_type} detail={error_detail}")

        enriched_item = dict(item)
        enriched_item["nl_sentences"] = sentences
        enriched_item["nl_status"] = status
        enriched_item["nl_error_type"] = error_type
        enriched_item["nl_error_detail"] = error_detail
        enriched_item["nl_expected_rows"] = expected_rows
        enriched_item["nl_generated_rows"] = len(sentences)
        enriched_item["nl_headers"] = headers
        enriched.append(enriched_item)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert markdown tables to one-row-one-sentence NL form using OpenAI."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().with_name("extracted_tables.json")),
        help="Path to input JSON with table markdown entries.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().with_name("extracted_tables_nl3.json")),
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of table entries to process.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Delay between API calls.")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    tables = load_json(input_path)

    if args.limit is not None:
        tables = tables[: args.limit]

    client = create_openai_client()
    enriched = enrich_tables_with_sentences(
        tables=tables,
        client=client,
        model=args.model,
        sleep_seconds=args.sleep_seconds,
    )

    save_json(output_path, enriched)
    ok_count = sum(1 for entry in enriched if entry.get("nl_status") == "ok")
    mismatch_count = sum(1 for entry in enriched if entry.get("nl_status") == "mismatch")
    error_count = sum(1 for entry in enriched if entry.get("nl_status") == "error")
    print(f"Saved {len(enriched)} entries to: {output_path}")
    print(f"Status summary -> ok: {ok_count}, mismatch: {mismatch_count}, error: {error_count}")


if __name__ == "__main__":
    main()
