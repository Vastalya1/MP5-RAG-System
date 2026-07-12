import builtins
import contextvars
import json
import logging
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


_LOGGING_CONFIGURED = False
_STD_STREAMS_CAPTURED = False
_PRINT_CAPTURED = False
_transaction_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("transaction_id", default="-")
_query_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("query_id", default="-")
_QUERY_LOG_LOCK = threading.Lock()

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(password|passwd|pwd|secret|token|api[_-]?key|authorization|cookie|session)",
    re.IGNORECASE,
)
_POSTGRES_DSN_PATTERN = re.compile(r"(postgres(?:ql)?://)([^:@/\s]+):([^@/\s]+)@", re.IGNORECASE)
_BEARER_PATTERN = re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.=:+/]+", re.IGNORECASE)
_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b")
_GENERIC_LONG_SECRET_PATTERN = re.compile(r"\b(?:tvly|nvapi|ck)-[A-Za-z0-9_\-]{10,}\b", re.IGNORECASE)
_ASSIGNMENT_PATTERN = re.compile(
    r"(?P<key>password|passwd|pwd|secret|token|api[_-]?key|authorization|cookie|session)"
    r"(?P<sep>\s*[:=]\s*)"
    r"(?P<value>[^,\s]+)",
    re.IGNORECASE,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _logs_dir() -> Path:
    path = _repo_root() / "Logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _query_logs_dir() -> Path:
    path = _logs_dir() / "queries"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _redact_text(text: str) -> str:
    if not text:
        return text

    redacted = text
    redacted = _POSTGRES_DSN_PATTERN.sub(r"\1\2:***@", redacted)
    redacted = _BEARER_PATTERN.sub(r"\1***", redacted)
    redacted = _OPENAI_KEY_PATTERN.sub("***", redacted)
    redacted = _GENERIC_LONG_SECRET_PATTERN.sub("***", redacted)
    redacted = _ASSIGNMENT_PATTERN.sub(lambda m: f"{m.group('key')}{m.group('sep')}***", redacted)
    return redacted


def _sanitize_value(key: str, value: Any) -> str:
    if value is None:
        return "null"
    if _SENSITIVE_KEY_PATTERN.search(key):
        return "***"

    text = str(value)
    text = _redact_text(text)
    if len(text) > 300:
        return f"{text[:297]}..."
    return text


def _serialize_for_query_log(key: str, value: Any, max_length: int = 4000) -> str:
    if value is None:
        return "null"
    if _SENSITIVE_KEY_PATTERN.search(key):
        return "***"

    if isinstance(value, str):
        text = value
    elif isinstance(value, (int, float, bool)):
        text = str(value)
    else:
        try:
            text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)

    text = _redact_text(text)
    if len(text) > max_length:
        return f"{text[: max_length - 3]}..."
    return text


def _generated_preview(value: Any, max_length: int = 220) -> str:
    preview = _serialize_for_query_log("generated_preview", value, max_length=max_length)
    return preview.replace("\n", "\\n")


def format_fields(**fields: Any) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={_sanitize_value(key, value)}")
    return " ".join(parts)


class TransactionIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.transaction_id = _transaction_id_var.get("-")
        return True


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return _redact_text(message)


class _StreamToLogger:
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message: str) -> int:
        if not message:
            return 0
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.logger.log(self.level, line)
        return len(message)

    def flush(self) -> None:
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.strip())
        self._buffer = ""


def configure_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    logs_dir = _logs_dir()
    log_path = logs_dir / "application.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = RedactingFormatter(
        "%(asctime)s | %(levelname)s | tx=%(transaction_id)s | %(name)s | %(message)s"
    )
    transaction_filter = TransactionIdFilter()

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(transaction_filter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(transaction_filter)
    root_logger.addHandler(console_handler)

    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = True
    logging.getLogger("uvicorn.error").propagate = True

    _LOGGING_CONFIGURED = True


def capture_standard_streams() -> None:
    global _STD_STREAMS_CAPTURED
    if _STD_STREAMS_CAPTURED:
        return

    configure_logging()
    stdio_logger = logging.getLogger("stdio")
    sys.stdout = _StreamToLogger(stdio_logger, logging.INFO)
    sys.stderr = _StreamToLogger(stdio_logger, logging.ERROR)
    _STD_STREAMS_CAPTURED = True


def capture_print() -> None:
    global _PRINT_CAPTURED
    if _PRINT_CAPTURED:
        return

    configure_logging()
    logger = logging.getLogger("print")

    def logged_print(*args: Any, **kwargs: Any) -> None:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        message = sep.join(str(arg) for arg in args)
        if end and end != "\n":
            message = f"{message}{end}"
        message = message.strip()
        if message:
            logger.info(message)

    builtins.print = logged_print
    _PRINT_CAPTURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def new_transaction_id(prefix: str = "txn") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def set_transaction_id(transaction_id: str) -> contextvars.Token:
    return _transaction_id_var.set(transaction_id)


def reset_transaction_id(token: contextvars.Token) -> None:
    _transaction_id_var.reset(token)


def current_transaction_id() -> str:
    return _transaction_id_var.get("-")


def set_query_id(query_id: str) -> contextvars.Token:
    return _query_id_var.set(query_id)


def reset_query_id(token: contextvars.Token) -> None:
    _query_id_var.reset(token)


def current_query_id() -> str:
    return _query_id_var.get("-")


def current_query_log_path() -> Path | None:
    query_id = current_query_id()
    if not query_id or query_id == "-":
        return None
    safe_query_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)
    return _query_logs_dir() / f"{safe_query_id}.log"


def _append_query_log_entry(
    level: str,
    logger_name: str,
    step: str,
    generated: Any = None,
    **fields: Any,
) -> None:
    log_path = current_query_log_path()
    if log_path is None:
        return

    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    field_lines: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        serialized = _serialize_for_query_log(key, value)
        if "\n" in serialized:
            field_lines.append(f"{key}:")
            field_lines.extend(f"  {line}" for line in serialized.splitlines())
        else:
            field_lines.append(f"{key}: {serialized}")

    generated_text = _serialize_for_query_log("generated", generated) if generated is not None else ""

    lines = [
        "=" * 100,
        f"time: {timestamp}",
        f"level: {level}",
        f"transaction_id: {current_transaction_id()}",
        f"query_id: {current_query_id()}",
        f"logger: {logger_name}",
        f"step: {step}",
    ]
    if field_lines:
        lines.append("details:")
        lines.extend(field_lines)
    if generated is not None:
        lines.append("generated:")
        lines.extend(generated_text.splitlines() or [""])
    lines.append("")

    with _QUERY_LOG_LOCK:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines))


def start_query_log(
    query_id: str,
    logger: logging.Logger | None = None,
    generated: Any = None,
    **fields: Any,
) -> contextvars.Token:
    token = set_query_id(query_id)
    log_query_step(logger or logging.getLogger("query"), "query_started", generated=generated, **fields)
    return token


def log_query_step(logger: logging.Logger, step: str, generated: Any = None, **fields: Any) -> None:
    preview = _generated_preview(generated) if generated is not None else None
    payload = format_fields(step=step, generated_preview=preview, **fields)
    logger.info(f"query_step{f' | {payload}' if payload else ''}")
    _append_query_log_entry("INFO", logger.name, step, generated=generated, **fields)


def log_query_error(logger: logging.Logger, step: str, generated: Any = None, **fields: Any) -> None:
    preview = _generated_preview(generated) if generated is not None else None
    payload = format_fields(step=step, generated_preview=preview, **fields)
    logger.error(f"query_step_failed{f' | {payload}' if payload else ''}")
    _append_query_log_entry("ERROR", logger.name, step, generated=generated, **fields)


def log_info(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = format_fields(**fields)
    logger.info(f"{event}{f' | {payload}' if payload else ''}")


def log_error(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = format_fields(**fields)
    logger.error(f"{event}{f' | {payload}' if payload else ''}")
