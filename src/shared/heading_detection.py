import re


STRUCTURED_HEADING_RE = re.compile(
    r"^(?:section|clause|part|chapter)\s+[A-Za-z0-9IVXLCM\.]+(?:[\)\.\:-])?\s+.+$",
    re.IGNORECASE,
)
NUMERIC_HEADING_RE = re.compile(r"^(?:def\.\s*)?\d+(?:\.\d+){0,3}[\)\.\:-]?\s+.+$", re.IGNORECASE)
LETTER_HEADING_RE = re.compile(r"^[A-Z][\)\.]\s+.+$")
CODE_TAG_HEADING_RE = re.compile(r"^.+\(Code-[A-Za-z0-9_-]+\)$", re.IGNORECASE)


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def is_probable_heading(line: str) -> bool:
    line = normalize_line(line)
    if not line:
        return False

    words = line.split()
    lower = line.lower()

    if len(words) > 14 or len(line) > 140:
        return False

    if re.match(r"^(?:[a-z]|[ivxlcdm]+)[\)\.]\s+", lower):
        return False

    if STRUCTURED_HEADING_RE.match(line):
        return True
    if NUMERIC_HEADING_RE.match(line):
        return True
    if LETTER_HEADING_RE.match(line) and len(words) <= 16:
        return True
    if CODE_TAG_HEADING_RE.match(line) and len(words) <= 16:
        return True

    if line.endswith((".", ";", ",")):
        return False

    if line.endswith(":") and len(words) <= 12:
        return True
    if line.isupper() and 1 < len(words) <= 10:
        return True
    if line.istitle() and len(words) <= 8 and not line.endswith("."):
        return True
    return False
