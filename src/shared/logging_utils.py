import builtins
import contextvars
import logging
import os
import re
import sys
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


_LOGGING_CONFIGURED = False
_STD_STREAMS_CAPTURED = False
_PRINT_CAPTURED = False
_transaction_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("transaction_id", default="-")

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


def log_info(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = format_fields(**fields)
    logger.info(f"{event}{f' | {payload}' if payload else ''}")


def log_error(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = format_fields(**fields)
    logger.error(f"{event}{f' | {payload}' if payload else ''}")
