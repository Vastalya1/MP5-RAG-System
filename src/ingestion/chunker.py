import os
import re
import json
from collections import Counter
from typing import List, Dict, Optional, Tuple
import pdfplumber
from transformers import AutoTokenizer

# Load tokenizer (match your embedding model)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Chunking defaults for all-MiniLM-L6-v2 style embedding pipelines.
DEFAULT_MAX_TOKENS = 240
DEFAULT_OVERLAP = 0.15

# Common keywords often used as section titles in policies.
SECTION_KEYWORDS = [
    "coverage", "exclusion", "exclusions", "claim", "claims",
    "definition", "definitions", "eligibility", "benefit", "benefits",
    "policy", "conditions", "waiting period", "preamble"
]

PAGE_NUMBER_RE = re.compile(r"^(?:page\s*)?\d+(?:\s*of\s*\d+)?$", re.IGNORECASE)
STRUCTURED_HEADING_RE = re.compile(
    r"^(?:section|clause|part|chapter)\s+[A-Za-z0-9IVXLCM\.]+(?:[\)\.\:-])?\s+.+$",
    re.IGNORECASE,
)
NUMERIC_HEADING_RE = re.compile(r"^(?:def\.\s*)?\d+(?:\.\d+){0,3}[\)\.\:-]?\s+.+$", re.IGNORECASE)
CLAUSE_ID_RE = re.compile(r"^(?:def\.\s*)?(?P<id>\d+(?:\.\d+){0,3})[\)\.\:-]?", re.IGNORECASE)


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _looks_like_toc_page(lines: List[str]) -> bool:
    if not lines:
        return False
    joined = " ".join(lines).lower()
    if "table of contents" in joined:
        return True

    toc_indicators = sum(
        1
        for line in lines
        if re.search(r"\b(?:clause|section)\s+no\b", line.lower())
        or re.search(r"\bpage\s+no\b", line.lower())
    )
    toc_rows = sum(1 for line in lines if re.search(r"(?:\.{2,}|\s)\d{1,3}$", line))
    return toc_indicators > 0 and toc_rows >= max(6, int(len(lines) * 0.35))


def _extract_pdf_pages(pdf_path: str) -> List[List[str]]:
    page_lines: List[List[str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [_normalize_line(line) for line in text.splitlines()]
            lines = [line for line in lines if line]
            page_lines.append(lines)
    return page_lines


def _remove_repeated_boilerplate(page_lines: List[List[str]]) -> List[List[str]]:
    if not page_lines:
        return page_lines

    frequency = Counter()
    for lines in page_lines:
        for line in set(lines):
            frequency[line] += 1

    threshold = max(3, int(len(page_lines) * 0.5))
    repeated_lines = {line for line, count in frequency.items() if count >= threshold}

    cleaned_pages: List[List[str]] = []
    for lines in page_lines:
        if _looks_like_toc_page(lines):
            continue

        filtered = [
            line for line in lines
            if line not in repeated_lines and not PAGE_NUMBER_RE.match(line)
        ]
        cleaned_pages.append(filtered)
    return cleaned_pages


def is_probable_heading(line: str) -> bool:
    line = _normalize_line(line)
    if not line:
        return False

    words = line.split()
    lower = line.lower()

    # Ignore long lines. In these policy PDFs, long lines are usually content, not headings.
    if len(words) > 14 or len(line) > 140:
        return False

    # Bullet/list items are usually not section headings.
    if re.match(r"^(?:[a-z]|[ivxlcdm]+)[\)\.]\s+", lower):
        return False

    if STRUCTURED_HEADING_RE.match(line):
        return True
    if NUMERIC_HEADING_RE.match(line):
        return True
    if line.endswith(":") and len(words) <= 12:
        return True
    if line.isupper() and 1 < len(words) <= 10:
        return True
    if line.istitle() and len(words) <= 8 and not line.endswith("."):
        return True
    if any(keyword in lower for keyword in SECTION_KEYWORDS) and len(words) <= 10:
        return True
    return False


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text and remove repeated boilerplate lines and TOC-like pages."""
    page_lines = _extract_pdf_pages(pdf_path)
    cleaned_pages = _remove_repeated_boilerplate(page_lines)

    text_content = []
    for lines in cleaned_pages:
        if not lines:
            continue
        text_content.append("\n".join(lines))
    return "\n\n".join(text_content)


def _extract_clause_id(heading: str) -> Optional[str]:
    heading = _normalize_line(heading)
    if not heading:
        return None

    match = CLAUSE_ID_RE.match(heading)
    if match:
        return match.group("id")

    structured_match = re.match(
        r"^(?:section|clause|part|chapter)\s+([A-Za-z0-9IVXLCM\.]+)",
        heading,
        re.IGNORECASE,
    )
    if structured_match:
        return structured_match.group(1)
    return None


def _split_sentences(text: str) -> List[str]:
    cleaned = _normalize_line(text)
    if not cleaned:
        return []

    # Keep chunk boundaries near sentence boundaries for better retrieval coherence.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", cleaned)
    if len(sentences) == 1:
        sentences = re.split(r";\s+", cleaned)
    return [_normalize_line(sentence) for sentence in sentences if _normalize_line(sentence)]


def _decode_tokens(tokens: List[int]) -> str:
    return _normalize_line(tokenizer.decode(tokens, skip_special_tokens=True))


def _chunk_section_text(text: str, max_tokens: int, overlap: float) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    overlap_tokens = max(1, int(max_tokens * overlap))
    hard_split_step = max(1, max_tokens - overlap_tokens)

    chunks: List[str] = []
    current_sentences: List[Tuple[str, int]] = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_token_count = len(sentence_tokens)

        if sentence_token_count == 0:
            continue

        if sentence_token_count > max_tokens:
            if current_sentences:
                chunks.append(" ".join(sent for sent, _ in current_sentences))
                current_sentences = []
                current_token_count = 0

            for index in range(0, sentence_token_count, hard_split_step):
                window = sentence_tokens[index:index + max_tokens]
                if window:
                    chunks.append(_decode_tokens(window))
            continue

        if current_sentences and current_token_count + sentence_token_count > max_tokens:
            chunks.append(" ".join(sent for sent, _ in current_sentences))

            carry: List[Tuple[str, int]] = []
            carry_tokens = 0
            for prev_sentence, prev_tokens in reversed(current_sentences):
                if carry_tokens >= overlap_tokens:
                    break
                carry.insert(0, (prev_sentence, prev_tokens))
                carry_tokens += prev_tokens

            current_sentences = carry + [(sentence, sentence_token_count)]
            current_token_count = carry_tokens + sentence_token_count
            continue

        current_sentences.append((sentence, sentence_token_count))
        current_token_count += sentence_token_count

    if current_sentences:
        chunks.append(" ".join(sent for sent, _ in current_sentences))

    return [_normalize_line(chunk) for chunk in chunks if _normalize_line(chunk)]


def _merge_small_chunks(chunks: List[Dict], max_tokens: int, min_tokens: int) -> List[Dict]:
    if not chunks:
        return []

    merged: List[Dict] = []
    index = 0

    while index < len(chunks):
        current = chunks[index].copy()
        current_tokens = len(tokenizer.encode(current["text"], add_special_tokens=False))

        while current_tokens < min_tokens and index + 1 < len(chunks):
            nxt = chunks[index + 1]
            if nxt["metadata"]["section_heading"] != current["metadata"]["section_heading"]:
                break
            if nxt["metadata"].get("clause_id") != current["metadata"].get("clause_id"):
                break

            next_tokens = len(tokenizer.encode(nxt["text"], add_special_tokens=False))
            if current_tokens + next_tokens > max_tokens:
                break

            current["text"] = f"{current['text']} {nxt['text']}".strip()
            current_tokens += next_tokens
            index += 1

        merged.append(current)
        index += 1

    return merged


def _build_sections(text: str) -> List[Tuple[str, Optional[str], str]]:
    lines = text.splitlines()
    sections: List[Tuple[str, Optional[str], str]] = []

    current_section = "General"
    current_clause_id = None
    buffer: List[str] = []

    for line in lines:
        normalized = _normalize_line(line)
        if not normalized:
            if buffer:
                sections.append((current_section, current_clause_id, " ".join(buffer)))
                buffer = []
            continue

        if is_probable_heading(normalized):
            if buffer:
                sections.append((current_section, current_clause_id, " ".join(buffer)))
                buffer = []
            current_section = normalized.rstrip(":")
            current_clause_id = _extract_clause_id(current_section)
            continue

        buffer.append(normalized)

    if buffer:
        sections.append((current_section, current_clause_id, " ".join(buffer)))

    return sections


def chunk_document(
    text: str,
    doc_name: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: float = DEFAULT_OVERLAP,
) -> List[Dict]:
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be between 0 (inclusive) and 1 (exclusive).")

    sections = _build_sections(text)
    raw_chunks: List[Dict] = []

    for section_heading, clause_id, section_text in sections:
        sentence_chunks = _chunk_section_text(section_text, max_tokens=max_tokens, overlap=overlap)
        for text_chunk in sentence_chunks:
            raw_chunks.append(
                {
                    "chunk_id": "",  # assigned after optional short-chunk merge
                    "text": text_chunk,
                    "metadata": {
                        "document_name": doc_name,
                        "section_heading": section_heading,
                        "clause_id": clause_id,
                    },
                }
            )

    min_tokens = max(40, int(max_tokens * 0.25))
    chunks = _merge_small_chunks(raw_chunks, max_tokens=max_tokens, min_tokens=min_tokens)

    for index, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"{doc_name}_{index}"
    return chunks


def chunk_pdf_files(file_paths: List[str], output_file: str = None) -> List[Dict]:
    """
    Process specific PDF files, chunk them, and return a list of chunks.
    Optionally save them to JSON.
    """
    all_chunks = []

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if not file_name.lower().endswith(".pdf"):
            continue

        print(f"Extracting text from {file_name}...")
        text = extract_text_from_pdf(file_path)
        if not text.strip():
            print(f"Skipped {file_name} - no extractable text")
            continue

        chunks = chunk_document(text, doc_name=file_name)
        all_chunks.extend(chunks)
        print(f"Processed {file_name} - {len(chunks)} chunks")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved all chunks to {output_file}")

    return all_chunks


def chunk_pdfs(input_folder: str, output_file: str = None) -> List[Dict]:
    """
    Process all PDFs in a folder, chunk them, and return a list of chunks.
    Optionally save them to JSON.
    """
    file_paths = [
        os.path.join(input_folder, file_name)
        for file_name in sorted(os.listdir(input_folder))
        if file_name.lower().endswith(".pdf")
    ]
    return chunk_pdf_files(file_paths, output_file=output_file)
