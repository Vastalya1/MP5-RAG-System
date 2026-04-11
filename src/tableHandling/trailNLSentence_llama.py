import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


SYSTEM_PROMPT = (
    "You are a data extraction specialist focused on high-accuracy insurance documentation."
)


USER_PROMPT_TEMPLATE = """Role: You are a data extraction specialist focused on high-accuracy insurance documentation.

Task: Convert the provided Markdown table into a list of natural language sentences.

Strict Constraints:
1. One Row, One Sentence: Each row in the table must correspond to exactly one full sentence.
2. Strictly generate only the sentences and nothing else.
3. Zero Hallucination/Summarization: Do not add information not present in the table.
4. Explicit Header References: Explicitly mention the table headers in every sentence.
5. Maintain Terminology: Use the exact technical terms for covers and values as written in the markdown.
6. Handling Multiple Values: If a header has multiple values, include all values.
7. Preserve row order from top to bottom.

Validation Requirements:
- The table has exactly {expected_rows} data rows (excluding header and separator).
- Return exactly {expected_rows} sentences.
- One sentence per line.

Markdown Input:
{markdown}

Example Format:
The [COVER NAME] is [VALUE] under [HEADER-1] and [VALUE] under [HEADER-2].
"""


def _parse_markdown_row(line: str) -> List[str]:
    """Parse one markdown table row into cells."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_separator_row(cells: List[str]) -> bool:
    """Return True if the row is markdown separator row like | --- | --- |."""
    if not cells:
        return False
    return all(re.fullmatch(r"[:\-\s]+", cell or "") is not None for cell in cells)


def parse_markdown_table(markdown: str) -> Tuple[List[str], List[List[str]]]:
    """Extract headers and data rows from markdown table."""
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
    """Normalize model output into a clean sentence list."""
    cleaned = text.replace("```", "").strip()
    if not cleaned:
        return []

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    sentences: List[str] = []

    for line in lines:
        line = re.sub(r"^\s*(?:\d+[\).\:-]\s*|[-*]\s+)", "", line).strip()
        if line:
            sentences.append(line)

    # Fallback if model returns one long paragraph
    if len(sentences) <= 1 and "\n" not in cleaned:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", cleaned)
        fallback = [re.sub(r"^\s*(?:\d+[\).\:-]\s*|[-*]\s+)", "", p).strip() for p in parts if p.strip()]
        if len(fallback) > len(sentences):
            sentences = fallback

    return sentences


def rebalance_sentence_count(sentences: List[str], expected_rows: int) -> List[str]:
    """
    Heuristically rebalance sentence count to expected_rows.
    Useful when model wraps one sentence into multiple lines.
    """
    if expected_rows <= 0:
        return []
    if not sentences:
        return []

    work = [s.strip() for s in sentences if s and s.strip()]

    # Merge lines until count matches
    while len(work) > expected_rows and len(work) > 1:
        merge_idx: Optional[int] = None

        # Prefer merging when previous line doesn't look sentence-complete
        for i in range(len(work) - 1):
            left = work[i]
            right = work[i + 1]
            if not re.search(r"[.!?]$", left) or re.match(r"^[a-z(,]", right):
                merge_idx = i
                break

        # Fallback: merge shortest adjacent pair
        if merge_idx is None:
            merge_idx = min(
                range(len(work) - 1),
                key=lambda i: len(work[i]) + len(work[i + 1]),
            )

        work[merge_idx] = f"{work[merge_idx].rstrip()} {work[merge_idx + 1].lstrip()}".strip()
        del work[merge_idx + 1]

    # Split long lines until count matches
    while len(work) < expected_rows and work:
        split_idx = max(range(len(work)), key=lambda i: len(work[i]))
        text = work[split_idx]
        split_point: Optional[int] = None

        for pattern in [r", and ", r" and ", r"; ", r", "]:
            matches = list(re.finditer(pattern, text))
            if matches:
                mid = len(text) / 2
                best = min(matches, key=lambda m: abs(m.start() - mid))
                split_point = best.start()
                break

        if split_point is None:
            break

        left = text[:split_point].strip().rstrip(",;")
        right = text[split_point:].strip().lstrip(",;").strip()
        if not left or not right:
            break

        work[split_idx:split_idx + 1] = [left, right]

    return work


def rows_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """Build markdown table string from headers and data rows."""
    if not headers:
        return ""
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = []
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        row_lines.append("| " + " | ".join(padded[: len(headers)]) + " |")
    return "\n".join([header_line, separator_line, *row_lines])


def _llama_generate_text(
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_seconds: int = 120,
) -> str:
    """Generate text from NVIDIA-hosted Llama model using OpenAI-compatible API."""
    combined_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    safe_max_tokens = min(max_tokens, 1024)
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": combined_prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=safe_max_tokens,
        stream=True,
        timeout=timeout_seconds,
    )

    parts: List[str] = []
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            parts.append(content)

    return "".join(parts).strip()


def generate_sentences_for_table(
    client: Any,
    model: str,
    markdown: str,
    expected_rows: int,
    max_tokens: int = 2048,
    max_attempts: int = 3,
    timeout_seconds: int = 120,
) -> Tuple[List[str], str]:
    """
    Generate one sentence per row from a markdown table.
    Returns (sentences, status) where status in {"ok", "mismatch", "error"}.
    """
    if not markdown.strip() or expected_rows <= 0:
        return [], "ok"

    prompt = USER_PROMPT_TEMPLATE.format(expected_rows=expected_rows, markdown=markdown)
    last_sentences: List[str] = []
    last_error: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        try:
            content = _llama_generate_text(
                client=client,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)}"
            print(f"Llama call failed (attempt {attempt}/{max_attempts}): {last_error[:180]}")
            time.sleep(0.8)
            continue

        sentences = normalize_sentence_lines(content)
        last_sentences = sentences

        if len(sentences) == expected_rows:
            return sentences, "ok"

        prompt = (
            f"Your previous output returned {len(sentences)} sentences, but required {expected_rows}.\n"
            f"Return exactly {expected_rows} sentences, one per line, no numbering, no extra text.\n\n"
            f"Markdown Input:\n{markdown}"
        )

    if last_sentences:
        balanced = rebalance_sentence_count(last_sentences, expected_rows)
        if len(balanced) == expected_rows:
            return balanced, "ok_rebalanced"
        return last_sentences, "mismatch"

    if last_error:
        print(f"Llama generation failed after retries: {last_error[:220]}")
    return [], "error"


def generate_sentences_chunked(
    client: Any,
    model: str,
    headers: List[str],
    data_rows: List[List[str]],
    chunk_size: int = 20,
    timeout_seconds: int = 120,
) -> Tuple[List[str], str]:
    """Fallback path: split large table into row chunks and process each chunk."""
    if not data_rows:
        return [], "ok"
    if not headers:
        return [], "error"

    all_sentences: List[str] = []
    for start in range(0, len(data_rows), chunk_size):
        chunk_rows = data_rows[start : start + chunk_size]
        markdown_chunk = rows_to_markdown(headers, chunk_rows)
        chunk_sentences, chunk_status = generate_sentences_for_table(
            client=client,
            model=model,
            markdown=markdown_chunk,
            expected_rows=len(chunk_rows),
            max_tokens=1536,
            max_attempts=2,
            timeout_seconds=timeout_seconds,
        )

        if chunk_status == "mismatch":
            chunk_sentences, chunk_status = generate_sentences_rowwise(
                client=client,
                model=model,
                headers=headers,
                data_rows=chunk_rows,
                max_tokens=384,
                max_attempts=2,
                timeout_seconds=timeout_seconds,
            )

        if chunk_status not in {"ok", "ok_rowwise", "ok_rebalanced"} or len(chunk_sentences) != len(chunk_rows):
            return all_sentences, "error"

        all_sentences.extend(chunk_sentences)

    return all_sentences, "ok_chunked"


def generate_sentences_rowwise(
    client: Any,
    model: str,
    headers: List[str],
    data_rows: List[List[str]],
    max_tokens: int = 384,
    max_attempts: int = 2,
    timeout_seconds: int = 120,
) -> Tuple[List[str], str]:
    """Fallback path: generate one sentence per row with per-row API calls."""
    sentences: List[str] = []
    if not data_rows:
        return sentences, "ok"

    header_text = " | ".join(headers)
    for row_idx, row in enumerate(data_rows, start=1):
        row_text = " | ".join(row)
        prompt = f"""Role: You are a data extraction specialist focused on high-accuracy insurance documentation.

Task: Convert one table row into exactly one full natural-language sentence.

Strict Constraints:
1. Output exactly one sentence.
2. Output only that sentence.
3. Do not hallucinate or summarize.
4. Explicitly reference relevant headers from this table in the sentence.
5. Preserve exact technical terms and values.

Headers:
{header_text}

Row {row_idx}:
{row_text}
"""

        sentence: Optional[str] = None
        for _ in range(max_attempts):
            try:
                content = _llama_generate_text(
                    client=client,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                )
                parsed = normalize_sentence_lines(content)
                if parsed:
                    sentence = " ".join(parsed).strip()
                    break
            except Exception as e:
                print(f"Llama row call failed: {type(e).__name__}: {str(e)[:160]}")
                time.sleep(0.5)
                continue

        if sentence is None:
            return sentences, "error"
        sentences.append(sentence)

    return sentences, "ok_rowwise"


def load_json(path: Path) -> List[Dict[str, Any]]:
    """Load list JSON file."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON in {path}, found: {type(payload).__name__}")
    return payload


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    """Write list JSON file with utf-8 and readable formatting."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert markdown tables to one-row-one-sentence NL form using NVIDIA Llama ChatQA."
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
        default=str(Path(__file__).resolve().with_name("extracted_tables_nl_llama.json")),
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nvidia/llama3-chatqa-1.5-8b",
        help="NVIDIA model name (default: nvidia/llama3-chatqa-1.5-8b).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of table entries to process.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Delay between API calls.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Per-request timeout in seconds (default: 120).",
    )
    args = parser.parse_args()

    if OpenAI is None:
        raise RuntimeError("openai package is required. Install with: pip install openai")

    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")

    api_key = (
        os.getenv("NVIDIA_API_KEY", "").strip()
        or os.getenv("NVIDIA_NIM_API_KEY", "").strip()
        or os.getenv("NGC_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError(
            "NVIDIA API key is missing. Set NVIDIA_API_KEY (or NVIDIA_NIM_API_KEY / NGC_API_KEY) in .env."
        )

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    tables = load_json(input_path)

    if args.limit is not None:
        tables = tables[: args.limit]

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )

    enriched: List[Dict[str, Any]] = []
    total = len(tables)
    for idx, item in enumerate(tables, start=1):
        markdown = str(item.get("markdown", "") or "")
        headers, data_rows = parse_markdown_table(markdown)
        expected_rows = len(data_rows)

        print(f"[{idx}/{total}] page={item.get('page_number')} rows={expected_rows} -> generating...")
        sentences, status = generate_sentences_for_table(
            client=client,
            model=args.model,
            markdown=markdown,
            expected_rows=expected_rows,
            timeout_seconds=args.timeout_seconds,
        )

        if status in {"mismatch", "error"} and expected_rows > 0:
            print(f"[{idx}/{total}] {status} detected -> chunked fallback...")
            chunked_sentences, chunked_status = generate_sentences_chunked(
                client=client,
                model=args.model,
                headers=headers,
                data_rows=data_rows,
                timeout_seconds=args.timeout_seconds,
            )
            if chunked_status in {"ok", "ok_chunked"} and len(chunked_sentences) == expected_rows:
                sentences = chunked_sentences
                status = "ok_chunked"
            else:
                print(f"[{idx}/{total}] chunked fallback failed -> row-wise fallback...")
                rowwise_sentences, rowwise_status = generate_sentences_rowwise(
                    client=client,
                    model=args.model,
                    headers=headers,
                    data_rows=data_rows,
                    timeout_seconds=args.timeout_seconds,
                )
                if rowwise_status in {"ok", "ok_rowwise"} and len(rowwise_sentences) == expected_rows:
                    sentences = rowwise_sentences
                    status = "ok_rowwise"
                else:
                    sentences = rowwise_sentences if rowwise_sentences else chunked_sentences
                    status = "error"

        enriched_item = dict(item)
        enriched_item["nl_sentences"] = sentences
        enriched_item["nl_status"] = status
        enriched_item["nl_expected_rows"] = expected_rows
        enriched_item["nl_generated_rows"] = len(sentences)
        enriched_item["nl_headers"] = headers
        enriched.append(enriched_item)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    save_json(output_path, enriched)
    ok_like = {"ok", "ok_rebalanced", "ok_chunked", "ok_rowwise"}
    ok_count = sum(1 for entry in enriched if entry.get("nl_status") in ok_like)
    mismatch_count = sum(1 for entry in enriched if entry.get("nl_status") == "mismatch")
    error_count = sum(1 for entry in enriched if entry.get("nl_status") == "error")
    print(f"Saved {len(enriched)} entries to: {output_path}")
    print(f"Status summary -> ok_like: {ok_count}, mismatch: {mismatch_count}, error: {error_count}")


if __name__ == "__main__":
    main()
