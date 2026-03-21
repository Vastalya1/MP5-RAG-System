"""
PDF Table Extraction

Extracts tables from PDF files, converts them to markdown format,
and saves results to JSON. Uses the same header/footer filtering 
approach as chunker.py.

Features:
- Handles tables spanning multiple pages
- Associates headings with tables
- Filters out boilerplate text (headers/footers)
"""

import re
import sys
import json
import os
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import Counter
from statistics import mean, pstdev


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from shared.heading_detection import normalize_line as _normalize_line


# Regex for page numbers (from chunker.py)
PAGE_NUMBER_RE = re.compile(r"^(?:page\s*)?\d+(?:\s*of\s*\d+)?$", re.IGNORECASE)

# Patterns indicating table continuation
CONTINUATION_PATTERNS = [
    re.compile(r"cont(?:inued|d)?\.?", re.IGNORECASE),
    re.compile(r"\(?(?:contd?|continued)\)?\.?", re.IGNORECASE),
]

# Multiple extraction settings improve robustness across grid-lined and borderless tables
TABLE_EXTRACTION_STRATEGIES = [
    {
        "name": "lattice_lines",
        "min_score": 0.42,
        "settings": {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 10,
            "intersection_tolerance": 3,
        },
    },
    {
        "name": "lattice_loose",
        "min_score": 0.40,
        "settings": {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 5,
            "join_tolerance": 6,
            "edge_min_length": 6,
            "intersection_tolerance": 5,
        },
    },
    {
        "name": "stream_text",
        "min_score": 0.52,
        "settings": {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "text_tolerance": 3,
            "intersection_tolerance": 5,
            "min_words_vertical": 2,
            "min_words_horizontal": 1,
        },
    },
]

# Light stopword set to detect paragraph-like rows split into pseudo columns
ROW_STOPWORDS = {
    "the", "and", "or", "of", "to", "in", "for", "with", "on", "by",
    "is", "are", "as", "at", "be", "this", "that", "from", "an", "a",
    "will", "shall", "under", "within", "into", "which", "where", "when",
}
def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp value to [lower, upper]."""
    return max(lower, min(value, upper))


def _trim_empty_edges(table: List[List]) -> List[List[str]]:
    """
    Normalize table and remove fully empty rows/columns at edges.
    This avoids scoring artifacts from extraction noise.
    """
    if not table:
        return []

    normalized_rows: List[List[str]] = []
    max_cols = 0
    for row in table:
        normalized_row = [_normalize_line(str(cell)) if cell else "" for cell in row]
        if any(normalized_row):
            normalized_rows.append(normalized_row)
            max_cols = max(max_cols, len(normalized_row))

    if not normalized_rows or max_cols == 0:
        return []

    padded_rows = [row + [""] * (max_cols - len(row)) for row in normalized_rows]

    non_empty_cols = []
    for col_idx in range(max_cols):
        if any(row[col_idx] for row in padded_rows):
            non_empty_cols.append(col_idx)

    if not non_empty_cols:
        return []

    compact_rows = [[row[col_idx] for col_idx in non_empty_cols] for row in padded_rows]
    return compact_rows


def _markdown_row_counts(markdown: str) -> Tuple[int, int]:
    """
    Return (table_lines, data_rows) for rendered markdown.

    - table_lines counts all lines that look like markdown rows, including the
      header row and the separator row.
    - data_rows counts body rows only, excluding header and separator.
    """
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 2:
        return len(table_lines), 0
    data_rows = max(0, len(table_lines) - 2)
    return len(table_lines), data_rows


def _debug_row_count_summary(
    page_num: int,
    table_index: int,
    raw_table: List[List],
    non_boilerplate_table: List[List],
    trimmed_table: List[List[str]],
    markdown: str,
) -> None:
    """
    Optional debug print for tracing how num_rows is derived.

    Enable with TABLE_DEBUG_ROWS=1 in the environment.
    """
    if os.getenv("TABLE_DEBUG_ROWS", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return

    markdown_lines, markdown_data_rows = _markdown_row_counts(markdown)
    print(
        f"  -> Row count debug [page {page_num} table {table_index}]: "
        f"raw_extracted_rows={len(raw_table)}, "
        f"after_boilerplate_filter={len(non_boilerplate_table)}, "
        f"after_trim_empty_edges={len(trimmed_table)} (= num_rows), "
        f"markdown_table_lines={markdown_lines}, "
        f"markdown_data_rows={markdown_data_rows}"
    )


def _table_quality_score(table: List[List], min_rows: int = 2, min_cols: int = 2) -> Tuple[float, Dict[str, float]]:
    """
    Compute a structural confidence score for a table candidate.
    Higher scores indicate higher likelihood that content is tabular.
    """
    compact_table = _trim_empty_edges(table)
    if not compact_table:
        return 0.0, {"rows": 0.0, "cols": 0.0, "density": 0.0}

    rows = len(compact_table)
    cols = max(len(row) for row in compact_table)
    padded_table = [row + [""] * (cols - len(row)) for row in compact_table]

    non_empty_counts = [sum(1 for cell in row if cell) for row in padded_table]
    total_cells = rows * cols
    non_empty_cells = sum(non_empty_counts)
    density = (non_empty_cells / total_cells) if total_cells > 0 else 0.0

    row_fill = [count / cols if cols > 0 else 0.0 for count in non_empty_counts]
    row_fill_mean = mean(row_fill) if row_fill else 0.0
    row_fill_std = pstdev(row_fill) if len(row_fill) > 1 else 0.0
    row_consistency = _clamp(1.0 - (row_fill_std / (row_fill_mean + 1e-6)))

    col_fill = []
    for col_idx in range(cols):
        col_non_empty = sum(1 for row in padded_table if row[col_idx])
        col_fill.append(col_non_empty / rows if rows > 0 else 0.0)
    col_coverage = (sum(1 for fill in col_fill if fill >= 0.35) / cols) if cols > 0 else 0.0

    text_cells = [cell for row in padded_table for cell in row if cell]
    unique_ratio = (
        len({cell.lower() for cell in text_cells}) / len(text_cells)
        if text_cells
        else 0.0
    )
    avg_cell_len = (
        sum(len(cell) for cell in text_cells) / len(text_cells)
        if text_cells
        else 0.0
    )
    long_cell_ratio = (
        sum(1 for cell in text_cells if len(cell) > 80) / len(text_cells)
        if text_cells
        else 0.0
    )
    lower_start_ratio = (
        sum(1 for cell in text_cells if re.match(r"^[a-z]", cell)) / len(text_cells)
        if text_cells
        else 0.0
    )

    structured_pattern = re.compile(
        r"^\s*(?:[\d,./%-]+|[$€£]?\s*[\d,]+(?:\.\d+)?|[A-Za-z]{1,4}-?\d+)\s*$"
    )
    structured_ratio = (
        sum(1 for cell in text_cells if structured_pattern.match(cell)) / len(text_cells)
        if text_cells
        else 0.0
    )

    header_row = padded_table[0] if padded_table else []
    header_cells = [cell for cell in header_row if cell]
    header_alpha_ratio = (
        sum(1 for cell in header_cells if re.search(r"[A-Za-z]", cell)) / len(header_cells)
        if header_cells
        else 0.0
    )

    sentence_like_rows = 0
    for row in padded_table:
        row_text = " ".join(cell for cell in row if cell).strip()
        tokens = re.findall(r"[A-Za-z']+", row_text.lower())
        stopword_count = sum(1 for token in tokens if token in ROW_STOPWORDS)
        if len(tokens) >= 8 and stopword_count >= 4 and len(row_text) >= 60:
            sentence_like_rows += 1
    sentence_like_ratio = sentence_like_rows / rows if rows > 0 else 0.0

    score = 0.0
    score += 0.18 * _clamp(rows / min_rows if min_rows > 0 else 1.0)
    score += 0.16 * _clamp(cols / min_cols if min_cols > 0 else 1.0)
    score += 0.24 * _clamp((density - 0.20) / 0.55)
    score += 0.14 * row_consistency
    score += 0.12 * col_coverage
    score += 0.08 * _clamp((unique_ratio - 0.10) / 0.60)
    score += 0.08 * _clamp(structured_ratio / 0.60)
    score += 0.04 * _clamp(header_alpha_ratio)

    if cols == 1:
        score -= 0.45 if rows < 6 else 0.18
    if density < 0.25 and rows < 5:
        score -= 0.20
    if long_cell_ratio > 0.45 and cols <= 2:
        score -= 0.25
    if avg_cell_len > 110:
        score -= 0.18
    if unique_ratio < 0.20 and len(text_cells) >= 10:
        score -= 0.10
    if cols >= 4 and lower_start_ratio > 0.55:
        score -= 0.30
    elif cols >= 4 and lower_start_ratio > 0.40:
        score -= 0.18
    if cols >= 4 and sentence_like_ratio > 0.35 and structured_ratio < 0.15:
        score -= 0.35
    elif cols >= 4 and sentence_like_ratio > 0.20 and structured_ratio < 0.10:
        score -= 0.15
    if rows >= 30 and cols >= 8 and structured_ratio < 0.08 and long_cell_ratio < 0.12:
        score -= 0.20

    metrics = {
        "rows": float(rows),
        "cols": float(cols),
        "density": density,
        "row_consistency": row_consistency,
        "col_coverage": col_coverage,
        "unique_ratio": unique_ratio,
        "avg_cell_len": avg_cell_len,
        "long_cell_ratio": long_cell_ratio,
        "lower_start_ratio": lower_start_ratio,
        "structured_ratio": structured_ratio,
        "header_alpha_ratio": header_alpha_ratio,
        "sentence_like_ratio": sentence_like_ratio,
    }
    return _clamp(score), metrics


def _table_acceptance_threshold(metrics: Dict[str, float]) -> float:
    """
    Dynamic threshold based on shape/content.
    Narrow tables require stronger evidence than wider dense tables.
    """
    rows = int(metrics.get("rows", 0))
    cols = int(metrics.get("cols", 0))
    density = metrics.get("density", 0.0)

    threshold = 0.46
    if cols <= 2:
        threshold = 0.58
    elif cols >= 5 and rows >= 4:
        threshold = 0.42

    if rows >= 10 and density >= 0.35:
        threshold = min(threshold, 0.40)

    return threshold


def _looks_like_toc_page(lines: List[str]) -> bool:
    """Check if page looks like a Table of Contents (from chunker.py)."""
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


def _get_boilerplate_lines(pdf_path: str) -> Set[str]:
    """
    Identify repeated header/footer lines across PDF pages.
    Lines appearing on 50%+ of pages are considered boilerplate.
    """
    page_lines: List[List[str]] = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = [_normalize_line(line) for line in text.splitlines()]
            lines = [line for line in lines if line]
            page_lines.append(lines)
    
    if not page_lines:
        return set()
    
    frequency = Counter()
    for lines in page_lines:
        for line in set(lines):
            frequency[line] += 1
    
    threshold = max(3, int(len(page_lines) * 0.5))
    boilerplate = {line for line, count in frequency.items() if count >= threshold}
    
    return boilerplate


def _is_boilerplate(text: str, boilerplate_lines: Set[str]) -> bool:
    """Check if text is boilerplate (header/footer/page number)."""
    if not text:
        return False
    
    normalized = _normalize_line(text)
    
    if normalized in boilerplate_lines:
        return True
    
    if PAGE_NUMBER_RE.match(normalized):
        return True
    
    return False


def _is_real_table(table: List[List], min_rows: int = 2, min_cols: int = 2) -> bool:
    """Check if extracted data is a real table (not fragmented text)."""
    score, metrics = _table_quality_score(table, min_rows=min_rows, min_cols=min_cols)
    if not table or metrics.get("rows", 0) < min_rows:
        return False

    threshold = _table_acceptance_threshold(metrics)
    return score >= threshold


def _bbox_overlap_ratio(bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> float:
    """Compute overlap ratio using intersection over smaller box area."""
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])

    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area1 = max(0.0, (bbox1[2] - bbox1[0])) * max(0.0, (bbox1[3] - bbox1[1]))
    area2 = max(0.0, (bbox2[2] - bbox2[0])) * max(0.0, (bbox2[3] - bbox2[1]))
    smaller_area = min(area1, area2)
    if smaller_area <= 0:
        return 0.0

    return inter_area / smaller_area


def _deduplicate_table_candidates(candidates: List[Dict[str, Any]], overlap_threshold: float = 0.85) -> List[Dict[str, Any]]:
    """
    Keep the best candidate per physical table region when multiple
    extraction strategies return the same table.
    """
    ranked = sorted(
        candidates,
        key=lambda c: (
            c.get("score", 0.0),
            c.get("num_rows", 0) * c.get("num_cols", 0),
        ),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    for cand in ranked:
        bbox = cand["bbox"]
        if any(_bbox_overlap_ratio(bbox, keep["bbox"]) >= overlap_threshold for keep in selected):
            continue
        selected.append(cand)

    return sorted(selected, key=lambda c: (c["bbox"][1], c["bbox"][0]))


def _extract_immediate_heading_from_text(
    text_before_table: str, 
    boilerplate: Set[str],
    max_blocks_to_check: int = 8
) -> Optional[str]:
    """
    Find the nearest same-page line/block that introduces the table.
    
    Args:
        text_before_table: Text content appearing before the table
        boilerplate: Set of boilerplate lines to ignore
        max_blocks_to_check: Number of nearby text blocks to examine
    
    Returns:
        The detected heading or None
    """
    if not text_before_table:
        return None
    
    lines = text_before_table.strip().split('\n')
    lines = [_normalize_line(line) for line in lines if _normalize_line(line)]
    
    # Filter out boilerplate lines
    lines = [line for line in lines if line not in boilerplate and not PAGE_NUMBER_RE.match(line)]
    
    if not lines:
        return None
    
    blocks = _build_context_blocks(lines)
    if not blocks:
        return None

    candidates = blocks[-max_blocks_to_check:] if len(blocks) > max_blocks_to_check else blocks
    fallback_heading = None
    encountered_table_content = False
    nearest_intro: Optional[Tuple[str, int]] = None
    nearest_heading: Optional[Tuple[str, int]] = None
    valid_rank = 0

    for block in reversed(candidates):
        normalized = block.strip()
        if not normalized:
            continue

        if _looks_like_table_rowish_block(normalized):
            encountered_table_content = True
            continue

        if _looks_like_introductory_block(normalized):
            if nearest_intro is None:
                nearest_intro = (normalized, valid_rank)
            valid_rank += 1
            continue

        if _looks_like_structural_heading(normalized):
            if not encountered_table_content and nearest_heading is None:
                nearest_heading = (normalized, valid_rank)
            if fallback_heading is None and not _is_weak_table_header(normalized):
                fallback_heading = normalized
            valid_rank += 1
            continue

        valid_rank += 1

    if nearest_intro and nearest_heading:
        intro_text, intro_rank = nearest_intro
        heading_text, heading_rank = nearest_heading
        if heading_rank < intro_rank:
            return heading_text
        if (
            heading_rank - intro_rank <= 2
            and _is_generic_explanatory_sentence(intro_text)
        ):
            return heading_text

    if nearest_intro:
        return nearest_intro[0]

    if nearest_heading:
        return nearest_heading[0]

    return fallback_heading


def _looks_like_table_item_line(line: str) -> bool:
    """
    Reject likely row/list content that happens to resemble a heading.

    Examples:
    - "203 US + coeliac node biopsy"
    - "19 MEDICAL CERTIFICATE"
    - "A. Uterine Artery Embolization and HIFU"
    """
    lower = line.lower()

    bullet_prefix = re.match(r"^(?:[a-z]|[ivxlcdm]+)[\)\.]\s+", lower)
    if bullet_prefix:
        if ":" in line or line.endswith(".") or " below" in lower or " as under" in lower:
            return False
        return True

    upper_bullet_prefix = re.match(r"^[A-Z]\.\s+", line)
    if upper_bullet_prefix:
        if ":" in line or line.endswith(".") or " below" in lower or " as under" in lower:
            return False
        return True

    if re.match(r"^\d+\s+[A-Za-z]", line):
        return "code" not in lower and ":" not in line and ")" not in line

    if re.match(r"^\d+\.\s+[A-Za-z]", line):
        return len(line.split()) > 3 and "code" not in lower and ":" not in line

    return False


def _looks_like_table_rowish_line(line: str) -> bool:
    """Detect lines that look like table rows or in-table headers."""
    normalized = line.strip()
    if not normalized:
        return False

    if _looks_like_reference_line(normalized):
        return True

    lower = normalized.lower()
    words = normalized.split()
    digit_groups = re.findall(r"\d+(?:\.\d+)?", normalized)

    if re.match(r"^\d+(?:\.\d+){0,2}[\)\.]\s+.+", normalized):
        return False

    if re.search(r"[:\-]", normalized) and len(words) <= 12 and not digit_groups:
        return False

    if normalized.isupper() and 2 <= len(words) <= 12 and not digit_groups:
        return False

    if len(words) <= 5 and normalized == normalized.title() and not digit_groups:
        return False

    if len(digit_groups) >= 2:
        return True

    if re.search(r"\b\d+(?:\.\d+)?\s*%", normalized):
        return True

    if re.match(r"^\d+\s+[A-Za-z]", normalized) and ":" not in normalized and len(words) <= 10:
        return True

    if normalized.isupper() and len(words) >= 5:
        return True

    if len(words) >= 6 and len(digit_groups) >= 1:
        short_ratio = sum(1 for word in words if len(word) <= 4) / len(words)
        if short_ratio >= 0.7 and not normalized.endswith((".", ":")):
            return True

    if _looks_like_table_item_line(normalized) and ":" not in normalized and " below" not in lower:
        return True

    return False


def _looks_like_table_rowish_block(block: str) -> bool:
    """Detect table content blocks that should not become table headings."""
    normalized = block.strip()
    if not normalized:
        return False

    if _looks_like_reference_line(normalized):
        return True

    if _looks_like_table_rowish_line(normalized):
        return True

    lines = [part.strip() for part in normalized.split(" | ") if part.strip()]
    if lines and all(_looks_like_table_rowish_line(part) for part in lines):
        return True

    numeric_spans = re.findall(r"\d+(?:\.\d+)?", normalized)
    if len(numeric_spans) >= 4:
        return True

    return False


def _looks_like_structural_heading(line: str) -> bool:
    """Detect standalone title-like lines that label the table's section."""
    raw = line.strip()
    normalized = raw.rstrip(":")
    if not normalized:
        return False

    if normalized.startswith("*"):
        return False

    if _looks_like_table_rowish_block(normalized):
        return False

    words = normalized.split()
    if len(words) < 2 or len(words) > 16:
        return False

    if normalized.endswith((".", ",", ";")):
        return False

    lower = normalized.lower()
    if lower.startswith("note:"):
        return False

    if re.match(r"^(?:[a-z]|[ivxlcdm]+)[\)\.]\s+", lower):
        return False

    if re.match(r"^\d+(?:\.\d+){0,2}[\)\.]\s+.+", normalized):
        return True

    digit_groups = re.findall(r"\d+(?:\.\d+)?", normalized)
    if digit_groups and not re.search(r"[:\-]", raw):
        return False

    if (":" in raw or "-" in raw) and len(words) <= 12:
        return True

    if normalized.isupper() and 1 < len(words) <= 12:
        return True

    if len(words) <= 5 and normalized == normalized.title() and not normalized.endswith("."):
        return True
    return False


def _looks_like_introductory_block(line: str) -> bool:
    """Detect the nearest same-page sentence that introduces the table."""
    normalized = line.strip()
    if not normalized:
        return False

    if normalized.startswith("*"):
        return False

    if _looks_like_table_rowish_block(normalized):
        return False

    lower = normalized.lower()
    if lower.startswith("note:"):
        return False

    words = normalized.split()
    if len(words) < 4 or len(words) > 40:
        return False

    if _looks_like_structural_heading(normalized):
        return False

    if re.search(r"[,;]", normalized):
        return True

    if normalized.endswith((".", ":")):
        return True

    if " below" in lower or " as under" in lower or " as below" in lower:
        return True

    if re.match(r"^(?:[a-z]|[ivxlcdm]+)[\)\.]\s+", lower):
        return len(words) >= 6

    lowercase_words = sum(1 for word in words if re.search(r"[a-z]", word))
    return lowercase_words >= max(3, len(words) // 2)


def _is_weak_table_header(line: str) -> bool:
    """Short title-like lines often come from the table header itself."""
    normalized = line.strip().rstrip(":")
    if not normalized:
        return False

    words = normalized.split()
    if len(words) < 2 or len(words) > 8:
        return False

    if re.match(r"^\d+(?:\.\d+){0,2}[\)\.]?\s+.+", normalized):
        return False

    if ":" in normalized:
        return False

    if normalized.isupper():
        return True

    return normalized == normalized.title()


def _is_generic_explanatory_sentence(line: str) -> bool:
    """
    Detect nearby prose that explains the section but does not directly label
    the table itself. These should not outrank a strong title immediately above.
    """
    normalized = line.strip()
    if not normalized:
        return False

    lower = normalized.lower()
    words = normalized.split()

    if len(words) < 6:
        return False

    if normalized.endswith(":"):
        return False

    if " below" in lower or " as under" in lower or " as below" in lower:
        return False

    if re.match(r"^(?:[a-z]|[ivxlcdm]+)[\)\.]\s+", lower):
        return False

    return normalized.endswith(".")


def _looks_like_reference_line(line: str) -> bool:
    """Reject footer/reference lines that are not semantic headings."""
    lower = line.lower()

    if re.search(r"\bpage\s*\d+\b", line, re.IGNORECASE):
        return True

    if "@" in line or "www." in lower or "http" in lower:
        return True

    if re.match(r"^(?:tel|telephone|phone|mobile|fax|email)\b", lower):
        return True

    digit_count = sum(1 for ch in line if ch.isdigit())
    if digit_count >= 6 and ("/" in line or "-" in line):
        return True

    return False


def _extract_last_heading_from_lines(lines: List[str], boilerplate: Set[str]) -> Optional[str]:
    """Track the most recent active heading across page boundaries."""
    filtered_lines = [
        _normalize_line(line)
        for line in lines
        if _normalize_line(line)
        and _normalize_line(line) not in boilerplate
        and not PAGE_NUMBER_RE.match(_normalize_line(line))
    ]

    for line in reversed(filtered_lines):
        normalized = line.strip()
        if _looks_like_structural_heading(normalized):
            return normalized

    return None


def _build_context_blocks(lines: List[str]) -> List[str]:
    """
    Merge wrapped lines into local context blocks while keeping headings,
    table rows, and table-intro lines separated.
    """
    if not lines:
        return []

    blocks: List[str] = []
    current: List[str] = []

    for line in lines:
        normalized = line.strip()
        if not normalized:
            continue

        if not current:
            current = [normalized]
            continue

        previous = current[-1]
        starts_new_block = (
            _looks_like_structural_heading(previous)
            or _looks_like_structural_heading(normalized)
            or _looks_like_table_rowish_line(previous)
            or _looks_like_table_rowish_line(normalized)
            or _looks_like_introductory_block(normalized)
            or previous.endswith((".", ":", ";"))
            or _looks_like_reference_line(previous)
        )

        if starts_new_block:
            blocks.append(" ".join(current).strip())
            current = [normalized]
        else:
            current.append(normalized)

    if current:
        blocks.append(" ".join(current).strip())

    return blocks


def _get_table_bounding_box(table_obj) -> Tuple[float, float, float, float]:
    """Get the bounding box (x0, y0, x1, y1) of a table."""
    return (table_obj.bbox[0], table_obj.bbox[1], table_obj.bbox[2], table_obj.bbox[3])


def _extract_text_above_table(page, table_bbox: Tuple[float, float, float, float]) -> str:
    """
    Extract text that appears above a table on the same page.
    
    Args:
        page: pdfplumber page object
        table_bbox: Bounding box of the table (x0, y0, x1, y1)
    
    Returns:
        Text appearing above the table
    """
    # Get area above the table
    x0, y0, x1, y1 = table_bbox
    
    # Define region above the table (full width, from top to table top)
    above_region = (0, 0, page.width, y0)
    
    try:
        cropped = page.within_bbox(above_region)
        text = cropped.extract_text() or ""
        return text
    except Exception:
        return ""


def _extract_numbers_from_table(table: List[List]) -> List[int]:
    """Extract numbered items from table cells to detect numbered list continuations."""
    numbers = []
    number_pattern = re.compile(r'^(\d+)[.\)\s]')
    
    for row in table:
        for cell in row:
            if cell:
                cell_text = str(cell).strip()
                match = number_pattern.match(cell_text)
                if match:
                    numbers.append(int(match.group(1)))
    
    return sorted(set(numbers))


def _extract_column_numbers(table: List[List]) -> Dict[int, List[int]]:
    """Extract leading numbers per column for robust continuation checks."""
    column_numbers: Dict[int, List[int]] = {}
    if not table:
        return column_numbers

    number_pattern = re.compile(r'^\s*(\d+)\b')
    num_cols = max(len(row) for row in table) if table else 0
    for col_idx in range(num_cols):
        values: List[int] = []
        for row in table:
            if col_idx >= len(row):
                continue
            cell = row[col_idx]
            if not cell:
                continue
            cell_text = str(cell).strip()
            match = number_pattern.match(cell_text)
            if match:
                values.append(int(match.group(1)))
        if values:
            column_numbers[col_idx] = sorted(set(values))

    return column_numbers


def _is_numbered_continuation(table1: List[List], table2: List[List]) -> bool:
    """
    Check if table2 is a numbered continuation of table1.
    
    For tables with numbered items (1, 2, 3... then 15, 16, 17...),
    checks if the numbering continues sequentially.
    """
    # First, try column-wise continuation (works for multi-column serial tables)
    cols1 = _extract_column_numbers(table1)
    cols2 = _extract_column_numbers(table2)
    common_cols = set(cols1.keys()) & set(cols2.keys())
    for col_idx in common_cols:
        c1 = cols1[col_idx]
        c2 = cols2[col_idx]
        if not c1 or not c2:
            continue
        gap = min(c2) - max(c1)
        if 0 < gap <= 3:
            return True

    # Fallback: global check across all cells
    nums1 = _extract_numbers_from_table(table1)
    nums2 = _extract_numbers_from_table(table2)
    if nums1 and nums2:
        max_num1 = max(nums1)
        min_num2 = min(nums2)
        if 0 < (min_num2 - max_num1) <= 3:
            return True

    return False


def _tables_are_similar(table1: List[List], table2: List[List], threshold: float = 0.8) -> bool:
    """
    Check if two tables have similar structure (likely continuation).
    
    Args:
        table1: First table data
        table2: Second table data  
        threshold: Similarity threshold (0 to 1)
    
    Returns:
        True if tables appear to be same structure
    """
    if not table1 or not table2:
        return False
    
    # Compare column count
    cols1 = max(len(row) for row in table1)
    cols2 = max(len(row) for row in table2)
    
    if cols1 != cols2:
        return False
    
    # Check for numbered list continuation (e.g., items 1-14 then 15-35)
    if _is_numbered_continuation(table1, table2):
        return True
    
    # Compare header structure if present
    header1 = table1[0] if table1 else []
    header2 = table2[0] if table2 else []
    
    # Check if headers are similar or if table2 has no real header (continuation)
    header1_text = [str(cell).strip().lower() if cell else "" for cell in header1]
    header2_text = [str(cell).strip().lower() if cell else "" for cell in header2]
    
    # Exact header match
    if header1_text == header2_text:
        return True
    
    # Check if second table's first row looks like data (no header words)
    header_words = {
        'sr', 'no', 'sl', 'name', 'description', 'amount', 'date',
        'particular', 'item', 'details', 'type', 'category', 'total'
    }

    # Token-based matching prevents false hits like 'no' in 'tenotomy'
    header2_has_header_words = False
    for cell in header2:
        if not cell:
            continue
        tokens = re.findall(r"[a-z]+", str(cell).lower())
        if any(token in header_words for token in tokens):
            header2_has_header_words = True
            break
    
    return False


def _is_continuation_table(
    page_text: str, 
    table_bbox: Tuple[float, float, float, float],
    prev_table: Optional[List[List]],
    current_table: List[List],
    page_height: float = 792  # Default letter size height
) -> bool:
    """
    Determine if a table is a continuation of a previous table.
    
    Args:
        page_text: Text of current page
        table_bbox: Bounding box of current table
        prev_table: Previous table data (from previous page)
        current_table: Current table data
        page_height: Height of the page
    
    Returns:
        True if this appears to be a continuation table
    """
    if prev_table is None:
        return False
    
    # Check for explicit continuation markers in page text
    for pattern in CONTINUATION_PATTERNS:
        if pattern.search(page_text[:500]):  # Check beginning of page
            return True
    
    # Check for numbered list continuation first (most reliable)
    if _is_numbered_continuation(prev_table, current_table):
        return True
    
    # Check if table starts in upper portion of page (common for continuations)
    # Using relative position - table starts in top 25% of the page
    relative_position = table_bbox[1] / page_height if page_height > 0 else 0
    if relative_position < 0.25:
        if _tables_are_similar(prev_table, current_table):
            return True
    
    return _tables_are_similar(prev_table, current_table)


def _merge_tables(table1: List[List], table2: List[List], skip_header: bool = True) -> List[List]:
    """
    Merge two tables into one.
    
    Args:
        table1: Base table
        table2: Table to append
        skip_header: Whether to skip the first row of table2 (if it's a duplicate header)
    
    Returns:
        Merged table
    """
    if not table1:
        return table2
    if not table2:
        return table1
    
    merged = list(table1)
    
    # Check if we should skip the header of table2
    if skip_header and len(table2) > 1:
        header1 = [str(cell).strip().lower() if cell else "" for cell in table1[0]]
        header2 = [str(cell).strip().lower() if cell else "" for cell in table2[0]]
        
        if header1 == header2:
            merged.extend(table2[1:])
        else:
            merged.extend(table2)
    else:
        merged.extend(table2)
    
    return merged


def _table_to_markdown(table: List[List[str]]) -> str:
    """Convert a table to markdown format."""
    if not table or not table[0]:
        return ""
    
    clean_table = []
    for row in table:
        clean_row = [_normalize_line(str(cell)) if cell else "" for cell in row]
        clean_table.append(clean_row)
    
    num_cols = max(len(row) for row in clean_table)
    
    for row in clean_table:
        while len(row) < num_cols:
            row.append("")
    
    col_widths = [3] * num_cols
    for row in clean_table:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    
    lines = []
    
    header = clean_table[0]
    header_line = "| " + " | ".join(
        cell.ljust(col_widths[i]) for i, cell in enumerate(header)
    ) + " |"
    lines.append(header_line)
    
    separator = "| " + " | ".join("-" * col_widths[i] for i in range(num_cols)) + " |"
    lines.append(separator)
    
    for row in clean_table[1:]:
        row_line = "| " + " | ".join(
            cell.ljust(col_widths[i]) for i, cell in enumerate(row)
        ) + " |"
        lines.append(row_line)
    
    return "\n".join(lines)


def _resolve_merged_cells(
    table: List[List[str]],
    fill_vertical: bool = True,
    fill_horizontal: bool = True
) -> List[List[str]]:
    """
    Heuristically expand merged cells into explicit values.

    - Horizontal merges: propagate obvious row-level text spans in value area.
    - Vertical merges: propagate only from strong row-span anchors, avoiding
      accidental carry-over from numeric rows above.
    """
    if not table:
        return []

    grid = _trim_empty_edges(table)
    if not grid:
        return []

    num_cols = max(len(row) for row in grid)
    resolved = [row + [""] * (num_cols - len(row)) for row in grid]

    if fill_horizontal:
        for row_idx, row in enumerate(resolved):
            non_empty_indices = [idx for idx, cell in enumerate(row) if cell.strip()]
            if not non_empty_indices:
                continue

            # Fill trailing blanks to the right when a row clearly has a merged span.
            # This handles headers/value bands like: "IMPERIAL PLUS PLAN |  |  "
            # and section rows like: "Out-patient benefits |  |  | ...".
            last_non_empty_idx = non_empty_indices[-1]
            has_trailing_blanks = last_non_empty_idx < (num_cols - 1)
            if has_trailing_blanks:
                likely_merged_row = (
                    num_cols > 2
                    and (
                        row_idx == 0
                        or (non_empty_indices[0] == 0 and len(non_empty_indices) >= 2)
                    )
                )
                if likely_merged_row:
                    anchor_val = row[last_non_empty_idx]
                    if anchor_val.strip():
                        for c in range(last_non_empty_idx + 1, num_cols):
                            if not resolved[row_idx][c].strip():
                                resolved[row_idx][c] = anchor_val

            # Pattern: descriptor in first column + one text value anchor in value area.
            # Example: "Bronchical Thermoplasty | Up to Sum Insured | ...blank..."
            if len(non_empty_indices) == 2 and non_empty_indices[0] == 0:
                anchor_idx = non_empty_indices[1]
                anchor_val = row[anchor_idx]
                if re.search(r"[A-Za-z]", anchor_val):
                    for c in range(anchor_idx + 1, num_cols):
                        if not resolved[row_idx][c].strip():
                            resolved[row_idx][c] = anchor_val

            # Fill blank gaps between two non-empty anchors (interior colspan)
            for left, right in zip(non_empty_indices, non_empty_indices[1:]):
                if right - left <= 1:
                    continue
                left_value = row[left]
                for c in range(left + 1, right):
                    if not row[c].strip():
                        resolved[row_idx][c] = left_value

    if fill_vertical:
        # Fill down only when previous row clearly represents a single merged text
        # value across the value-area columns (all non-empty values equal).
        for row_idx in range(1, len(resolved)):
            current_row = resolved[row_idx]
            current_values = current_row[1:] if num_cols > 1 else []
            if any(cell.strip() for cell in current_values):
                continue

            prev_row = resolved[row_idx - 1]
            prev_values = [cell for cell in (prev_row[1:] if num_cols > 1 else []) if cell.strip()]
            if not prev_values:
                continue

            anchor = prev_values[0]
            is_uniform_prev = all(cell == anchor for cell in prev_values)
            anchor_is_text = bool(re.search(r"[A-Za-z]", anchor))
            if not (is_uniform_prev and anchor_is_text):
                continue

            for c in range(1, num_cols):
                resolved[row_idx][c] = anchor

    return resolved


def _table_to_markdown_resolved(table: List[List[str]]) -> str:
    """Convert table to markdown after resolving merged cells."""
    resolved = _resolve_merged_cells(table)
    return _table_to_markdown(resolved)


def extract_tables_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract all tables from a PDF file with multi-page support and heading detection.
    
    Features:
    - Detects and merges tables spanning multiple pages
    - Associates headings with each table
    - Filters out boilerplate content
    """
    print(f"Processing: {pdf_path}")
    
    print("Identifying header/footer patterns...")
    boilerplate = _get_boilerplate_lines(pdf_path)
    print(f"Found {len(boilerplate)} boilerplate patterns to ignore")
    
    tables_data = []
    pending_table = None  # Table data being accumulated across pages
    pending_info = None   # Metadata for pending table
    active_heading = None
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}")
        
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            page_lines = [_normalize_line(line) for line in page_text.splitlines() if _normalize_line(line)]
            page_last_heading = _extract_last_heading_from_lines(page_lines, boilerplate)
            
            if _looks_like_toc_page(page_lines):
                print(f"Page {page_num}: Skipped (Table of Contents)")
                continue
            
            page_candidates: List[Dict[str, Any]] = []
            for strategy in TABLE_EXTRACTION_STRATEGIES:
                strategy_name = strategy["name"]
                try:
                    table_finder = page.debug_tablefinder(strategy["settings"])
                    found_tables = table_finder.tables
                except Exception as e:
                    print(f"Page {page_num}: Strategy '{strategy_name}' error - {e}")
                    continue

                for table_obj in found_tables:
                    table = table_obj.extract()
                    if not table:
                        continue

                    # Filter boilerplate rows
                    non_boilerplate_table = []
                    for row in table:
                        row_text = " ".join(str(cell) if cell else "" for cell in row)
                        if not _is_boilerplate(row_text, boilerplate):
                            non_boilerplate_table.append(row)

                    filtered_table = _trim_empty_edges(non_boilerplate_table)
                    if not filtered_table:
                        continue

                    score, metrics = _table_quality_score(filtered_table)
                    threshold = max(strategy["min_score"], _table_acceptance_threshold(metrics))
                    if score < threshold:
                        continue

                    table_bbox = _get_table_bounding_box(table_obj)
                    page_candidates.append({
                        "strategy": strategy_name,
                        "score": score,
                        "bbox": table_bbox,
                        "raw_table": table,
                        "non_boilerplate_table": non_boilerplate_table,
                        "table": filtered_table,
                        "num_rows": int(metrics.get("rows", len(filtered_table))),
                        "num_cols": int(metrics.get("cols", max(len(row) for row in filtered_table))),
                    })

            selected_candidates = _deduplicate_table_candidates(page_candidates)

            if not selected_candidates:
                # If no tables on this page but we have a pending table, finalize it
                if pending_table is not None:
                    pending_info["markdown"] = _table_to_markdown_resolved(pending_table)
                    pending_info["num_rows"] = len(pending_table)
                    pending_info["end_page"] = pending_info.get("end_page", pending_info["page_number"])
                    tables_data.append(pending_info)
                    print(f"  -> Finalized multi-page table: {pending_info['num_rows']} rows")
                    pending_table = None
                    pending_info = None
                continue
            
            table_count = 0
            for candidate in selected_candidates:
                filtered_table = candidate["table"]
                raw_table = candidate.get("raw_table", filtered_table)
                non_boilerplate_table = candidate.get("non_boilerplate_table", filtered_table)
                table_count += 1
                table_bbox = candidate["bbox"]
                page_height = page.height
                
                # Check if this is a continuation of a previous table
                is_continuation = _is_continuation_table(
                    page_text, 
                    table_bbox, 
                    pending_table,
                    filtered_table,
                    page_height
                )
                
                if is_continuation and pending_table is not None:
                    # Merge with pending table
                    pending_table = _merge_tables(pending_table, filtered_table)
                    pending_info["end_page"] = page_num
                    pending_info["spans_pages"] = True
                    print(f"Page {page_num}, Table {table_count}: Merged with previous (continuation)")
                    continue
                
                # If we have a pending table and this isn't a continuation, finalize it
                if pending_table is not None:
                    pending_info["markdown"] = _table_to_markdown_resolved(pending_table)
                    pending_info["num_rows"] = len(pending_table)
                    tables_data.append(pending_info)
                    print(f"  -> Finalized multi-page table: {pending_info['num_rows']} rows")
                    pending_table = None
                    pending_info = None
                
                # Extract heading for this table
                text_above = _extract_text_above_table(page, table_bbox)
                immediate_heading = _extract_immediate_heading_from_text(text_above, boilerplate)

                if immediate_heading:
                    heading = immediate_heading
                else:
                    relative_top = table_bbox[1] / page_height if page_height > 0 else 0
                    heading = active_heading if relative_top < 0.20 else None
                
                # Start new table
                table_info = {
                    "page_number": page_num,
                    "table_index": table_count,
                    "num_rows": len(filtered_table),
                    "num_cols": max(len(row) for row in filtered_table) if filtered_table else 0,
                    "heading": heading,
                    "spans_pages": False,
                    "markdown": _table_to_markdown_resolved(filtered_table),
                }
                _debug_row_count_summary(
                    page_num=page_num,
                    table_index=table_count,
                    raw_table=raw_table,
                    non_boilerplate_table=non_boilerplate_table,
                    trimmed_table=filtered_table,
                    markdown=table_info["markdown"],
                )
                
                # Check if table might continue to next page
                # Using relative position - table ends in bottom 15% of page
                relative_bottom = table_bbox[3] / page_height if page_height > 0 else 0
                if relative_bottom > 0.85:  # Table near bottom
                    pending_table = filtered_table
                    pending_info = table_info
                    print(
                        f"Page {page_num}, Table {table_count}: "
                        f"{table_info['num_rows']} rows x {table_info['num_cols']} cols "
                        f"(may continue) [{candidate['strategy']} score={candidate['score']:.2f}]"
                    )
                    if heading:
                        print(f"  -> Heading: {heading[:60]}...")
                else:
                    tables_data.append(table_info)
                    print(
                        f"Page {page_num}, Table {table_count}: "
                        f"{table_info['num_rows']} rows x {table_info['num_cols']} cols "
                        f"[{candidate['strategy']} score={candidate['score']:.2f}]"
                    )
                    if heading:
                        print(f"  -> Heading: {heading[:60]}...")
            if page_last_heading:
                active_heading = page_last_heading
        
        # Finalize any remaining pending table
        if pending_table is not None:
            pending_info["markdown"] = _table_to_markdown_resolved(pending_table)
            pending_info["num_rows"] = len(pending_table)
            tables_data.append(pending_info)
            print(f"  -> Finalized table: {pending_info['num_rows']} rows")
    
    return tables_data


def save_tables_to_json(tables_data: List[Dict], output_path: str):
    """Save extracted tables to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tables_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(tables_data)} tables to: {output_path}")


def main():
    """Main entry point."""
    base_dir = Path(__file__).resolve().parent
    pdf_path = base_dir / "test2.pdf"
    output_path = base_dir / "extracted_tables2.json"
    
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return
    
    print("="*60)
    print("PDF TABLE EXTRACTION (with multi-page & heading support)")
    print("="*60)
    
    tables_data = extract_tables_from_pdf(str(pdf_path))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tables extracted: {len(tables_data)}")
    
    if tables_data:
        total_rows = sum(t["num_rows"] for t in tables_data)
        multi_page_count = sum(1 for t in tables_data if t.get("spans_pages", False))
        tables_with_headings = sum(1 for t in tables_data if t.get("heading"))
        
        print(f"Total rows: {total_rows}")
        print(f"Multi-page tables: {multi_page_count}")
        print(f"Tables with headings: {tables_with_headings}")
        
        save_tables_to_json(tables_data, str(output_path))
        
        print("\n" + "="*60)
        print("SAMPLE OUTPUT (First Table)")
        print("="*60)
        first_table = tables_data[0]
        if first_table.get("heading"):
            print(f"Heading: {first_table['heading']}")
        if first_table.get("spans_pages"):
            print(f"Pages: {first_table['page_number']} - {first_table.get('end_page', first_table['page_number'])}")
        print(first_table["markdown"])
    else:
        print("No tables found.")


if __name__ == "__main__":
    main()
