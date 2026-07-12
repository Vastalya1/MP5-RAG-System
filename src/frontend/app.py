from fastapi import FastAPI, File, UploadFile, Form, Request, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from starlette.middleware.sessions import SessionMiddleware
import shutil
import os
from pathlib import Path
import chromadb
import time
import hashlib
import hmac
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import secrets
from passlib.context import CryptContext
from dotenv import load_dotenv
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.chroma_config import get_personal_collection_name, get_shared_collection_name
from shared.logging_utils import (
    capture_print,
    current_transaction_id,
    get_logger,
    log_error,
    log_info,
    log_query_error,
    log_query_step,
    new_transaction_id,
    reset_query_id,
    reset_transaction_id,
    set_transaction_id,
    start_query_log,
)
# from queryRewriter.rewriting import QueryRewriter
from queryRewriter.rewriting_Chatgpt import QueryRewriter
from retriever.retrival import retrivalModel
# from retriever.reranking_mistral import ChunkReranker
from retriever.reranking_Chatgpt import ChunkReranker
# from output.answerGeneration_mistral import AnswerGenerator
from output.answerGeneration_Chatgpt import AnswerGenerator
# Import LangGraph orchestrator
# from orchestration.orchestrator import QueryOrchestrator, create_orchestrator
from orchestration.orchestrator_Chatgpt import QueryOrchestrator, create_orchestrator
# Import your ingestion pipeline and other necessary components
# from ingestion.ingestionPipeline import IngestionPipeline
from ingestion.ingestionPipeline_Chatgpt import IngestionPipeline
from tavily_fallback.tavily_client import TavilySearchClient
from tavily_fallback.tavily_service import TavilyService


# Load environment variables from .env (if present)
load_dotenv()
capture_print()
logger = get_logger(__name__)

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

app = FastAPI()


@app.middleware("http")
async def log_request_middleware(request: Request, call_next):
    transaction_id = request.headers.get("x-transaction-id") or new_transaction_id("req")
    request.state.transaction_id = transaction_id
    token = set_transaction_id(transaction_id)
    start = time.perf_counter()
    log_info(
        logger,
        "request_started",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else None,
    )
    try:
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Transaction-Id"] = transaction_id
        log_info(
            logger,
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log_error(
            logger,
            "request_failed",
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        logger.exception("request_exception")
        raise
    finally:
        reset_transaction_id(token)

# Session middleware for login state
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret-change-me")
SESSION_MAX_AGE = int(os.getenv("SESSION_MAX_AGE", "1800"))
SESSION_HTTPS_ONLY = os.getenv("SESSION_HTTPS_ONLY", "false").lower() in {"1", "true", "yes"}
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    max_age=SESSION_MAX_AGE,
    https_only=SESSION_HTTPS_ONLY,
    same_site="strict",
)

# Setup template and static directories with absolute paths
templates = Jinja2Templates(directory=str(BASE_DIR / "src" / "frontend" / "templates"))
app.mount("/static", 
          StaticFiles(directory=str(BASE_DIR / "src" / "frontend" / "static")), 
          name="static")

# Ensure upload directory exists
UPLOAD_DIR = BASE_DIR / "dataset" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PERSONAL_UPLOAD_DIR = BASE_DIR / "dataset" / "user_uploads"
PERSONAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", "")
MAX_FAILED_ATTEMPTS = int(os.getenv("MAX_FAILED_ATTEMPTS", "5"))
LOCKOUT_MINUTES = int(os.getenv("LOCKOUT_MINUTES", "15"))
PASSWORD_MIN_LENGTH = int(os.getenv("PASSWORD_MIN_LENGTH", "8"))
PWD_CONTEXT = CryptContext(schemes=["argon2"], deprecated="auto")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DISABLE_CSRF = os.getenv("DISABLE_CSRF", "false").lower() in {"1", "true", "yes"}
CHROMA_SHARED_COLLECTION_NAME = get_shared_collection_name()
CHROMA_CLOUD_TENANT = os.getenv("CHROMA_CLOUD_TENANT", "a92961b0-ea65-4a82-a7ad-321a4baaaa60")
CHROMA_CLOUD_DATABASE = os.getenv("CHROMA_CLOUD_DATABASE", "Major-Project")

_retriever_instance: retrivalModel | None = None
_rewriter_instance: QueryRewriter | None = None
_reranker_instance: ChunkReranker | None = None
_answer_generator_instance: AnswerGenerator | None = None
_orchestrator_instance: QueryOrchestrator | None = None
_chroma_client_instance = None

# Flag to enable/disable orchestration (set to True to use LangGraph orchestration)
USE_ORCHESTRATION = os.getenv("USE_ORCHESTRATION", "true").lower() in {"1", "true", "yes"}

def _get_retriever() -> retrivalModel:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = retrivalModel()
    return _retriever_instance

def _get_rewriter() -> QueryRewriter:
    global _rewriter_instance
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for query rewriting.")
    if _rewriter_instance is None:
        _rewriter_instance = QueryRewriter(OPENAI_API_KEY)
    return _rewriter_instance

def _get_reranker() -> ChunkReranker:
    global _reranker_instance
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for reranking.")
    if _reranker_instance is None:
        _reranker_instance = ChunkReranker(OPENAI_API_KEY)
    return _reranker_instance

def _get_answer_generator() -> AnswerGenerator:
    global _answer_generator_instance
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for answer generation.")
    if _answer_generator_instance is None:
        _answer_generator_instance = AnswerGenerator(OPENAI_API_KEY)
    return _answer_generator_instance

def _get_orchestrator() -> QueryOrchestrator:
    """Get or create the LangGraph query orchestrator instance."""
    global _orchestrator_instance
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for orchestration.")
    if _orchestrator_instance is None:
        _orchestrator_instance = create_orchestrator(
            api_key=OPENAI_API_KEY,
            rewriter=_get_rewriter(),
            retriever=_get_retriever(),
            reranker=_get_reranker(),
            answer_generator=_get_answer_generator()
        )
    return _orchestrator_instance

def _get_chroma_client():
    global _chroma_client_instance
    if _chroma_client_instance is None:
        api_key = os.getenv("CHROMA_CLOUD_API_KEY")
        if not api_key:
            raise RuntimeError("CHROMA_CLOUD_API_KEY is required for Chroma access.")
        _chroma_client_instance = chromadb.CloudClient(
            api_key=api_key,
            tenant=CHROMA_CLOUD_TENANT,
            database=CHROMA_CLOUD_DATABASE,
        )
        log_info(
            logger,
            "chroma_client_initialized",
            tenant=CHROMA_CLOUD_TENANT,
            database=CHROMA_CLOUD_DATABASE,
        )
    return _chroma_client_instance

def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000,
    ).hex()

def _hash_password_strong(password: str) -> str:
    return PWD_CONTEXT.hash(password)

def _verify_password_strong(password: str, password_hash: str) -> bool:
    return PWD_CONTEXT.verify(password, password_hash)

def _validate_password(password: str) -> tuple[bool, str]:
    if len(password) < PASSWORD_MIN_LENGTH:
        return False, f"Password must be at least {PASSWORD_MIN_LENGTH} characters."
    if not any(ch.islower() for ch in password):
        return False, "Password must include a lowercase letter."
    if not any(ch.isupper() for ch in password):
        return False, "Password must include an uppercase letter."
    if not any(ch.isdigit() for ch in password):
        return False, "Password must include a digit."
    return True, ""

ADMIN_PASSWORDS = {
    "Prachi": "pk2026",
    "Sia": "s2026",
    "Akshada": "ak2026",
    "Vastalya": "avs2026",
}

@contextmanager
def _db_conn():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL must be set for Postgres access.")
    log_info(logger, "db_transaction_open")
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
        log_info(logger, "db_transaction_commit")
    except Exception:
        conn.rollback()
        log_error(logger, "db_transaction_rollback")
        logger.exception("db_transaction_exception")
        raise
    finally:
        conn.close()
        log_info(logger, "db_connection_closed")

def _get_user_record(username: str) -> dict | None:
    with _db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT username, role, salt, password_hash, password_algo,
                       failed_attempts, locked_until
                FROM users WHERE username = %s
                """,
                (username,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            record = {
                "username": row[0],
                "role": row[1],
                "salt": row[2],
                "password_hash": row[3],
                "password_algo": row[4],
                "failed_attempts": row[5],
                "locked_until": row[6],
            }
            if record["locked_until"] and record["locked_until"].tzinfo is None:
                record["locked_until"] = record["locked_until"].replace(tzinfo=timezone.utc)
            return record

def _verify_user(username: str, password: str) -> dict | None:
    record = _get_user_record(username)
    if not record:
        return None
    password_algo = record.get("password_algo") or "pbkdf2_sha256"
    if password_algo == "argon2":
        if _verify_password_strong(password, record["password_hash"]):
            return {"username": username, "role": record["role"]}
        return None

    expected = record["password_hash"]
    actual = _hash_password(password, record["salt"])
    if hmac.compare_digest(expected, actual):
        # Upgrade legacy hashes on successful login.
        new_hash = _hash_password_strong(password)
        with _db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE users
                    SET password_hash = %s, password_algo = %s, salt = NULL
                    WHERE username = %s
                    """,
                    (new_hash, "argon2", username),
                )
        return {"username": username, "role": record["role"]}
    return None

def _create_user(username: str, password: str) -> tuple[bool, str]:
    if username in ADMIN_PASSWORDS:
        return False, "Username is reserved."
    valid, message = _validate_password(password)
    if not valid:
        return False, message
    password_hash = _hash_password_strong(password)
    with _db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM users WHERE username = %s",
                (username,),
            )
            if cursor.fetchone():
                return False, "Username already exists."
            cursor.execute(
                """
                INSERT INTO users (username, role, salt, password_hash, password_algo)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (username, "user", None, password_hash, "argon2"),
            )
    return True, "Account created. You can sign in now."

def _ensure_csrf_token(request: Request) -> str:
    token = request.session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        request.session["csrf_token"] = token
    return token

def _require_csrf(request: Request, csrf_token: str | None) -> None:
    if DISABLE_CSRF:
        return
    if not csrf_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")
    session_token = request.session.get("csrf_token")
    if not session_token or not hmac.compare_digest(session_token, csrf_token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token")

def _record_audit_event(
    username: str | None,
    role: str | None,
    event_type: str,
    request: Request,
    resource_type: str | None = None,
    resource_id: str | None = None,
    metadata: dict | None = None,
) -> None:
    with _db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit_events
                (username, role, event_type, resource_type, resource_id, metadata, ip, user_agent)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    username,
                    role,
                    event_type,
                    resource_type,
                    resource_id,
                    psycopg2.extras.Json(metadata or {}),
                    request.client.host if request.client else None,
                    request.headers.get("user-agent"),
                ),
            )

def _init_db() -> None:
    with _db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    salt TEXT,
                    password_hash TEXT NOT NULL,
                    password_algo TEXT DEFAULT 'pbkdf2_sha256',
                    failed_attempts INTEGER DEFAULT 0,
                    locked_until TIMESTAMPTZ,
                    last_login_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                "ALTER TABLE users ALTER COLUMN salt DROP NOT NULL"
            )
            cursor.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS password_algo TEXT DEFAULT 'pbkdf2_sha256'"
            )
            cursor.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS failed_attempts INTEGER DEFAULT 0"
            )
            cursor.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS locked_until TIMESTAMPTZ"
            )
            cursor.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMPTZ"
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS uploaded_files (
                    id BIGSERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    username TEXT,
                    size_bytes BIGINT,
                    uploaded_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS uploaded_files_scope_idx
                ON uploaded_files (scope, username)
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id BIGSERIAL PRIMARY KEY,
                    username TEXT,
                    role TEXT,
                    event_type TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id TEXT,
                    metadata JSONB,
                    ip TEXT,
                    user_agent TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS query_history (
                    id BIGSERIAL PRIMARY KEY,
                    username TEXT NOT NULL,
                    role TEXT,
                    scope TEXT,
                    question TEXT NOT NULL,
                    answer TEXT,
                    justification TEXT,
                    sources JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS query_history_user_idx
                ON query_history (username, created_at DESC)
                """
            )
            for username, password in ADMIN_PASSWORDS.items():
                salt = f"{username}-salt"
                password_hash = _hash_password_strong(password)
                cursor.execute(
                    """
                    INSERT INTO users (username, role, salt, password_hash, password_algo)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (username) DO UPDATE
                    SET role = EXCLUDED.role,
                        salt = EXCLUDED.salt,
                        password_hash = EXCLUDED.password_hash,
                        password_algo = EXCLUDED.password_algo
                    """,
                    (username, "admin", None, password_hash, "argon2"),
                )

@app.on_event("startup")
def startup_event():
    _init_db()
    log_info(logger, "application_started", orchestration_enabled=USE_ORCHESTRATION)

def _get_current_user(request: Request) -> dict | None:
    return request.session.get("user")

def _require_auth(request: Request) -> dict:
    user = _get_current_user(request)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user

def _require_admin(request: Request) -> dict:
    user = _require_auth(request)
    if user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user

def _list_pdfs(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    return sorted([p.name for p in folder.glob("*.pdf") if p.is_file()])

def _safe_file_path(base_dir: Path, filename: str) -> Path:
    target = (base_dir / filename).resolve()
    if base_dir.resolve() not in target.parents or not target.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return target

def _list_uploaded_files(scope: str, username: str | None = None) -> list[dict]:
    with _db_conn() as conn:
        with conn.cursor() as cursor:
            if username:
                cursor.execute(
                    """
                    SELECT filename, username, size_bytes, uploaded_at
                    FROM uploaded_files
                    WHERE scope = %s AND username = %s
                    ORDER BY uploaded_at DESC
                    """,
                    (scope, username),
                )
            else:
                cursor.execute(
                    """
                    SELECT filename, username, size_bytes, uploaded_at
                    FROM uploaded_files
                    WHERE scope = %s
                    ORDER BY uploaded_at DESC
                    """,
                    (scope,),
                )
            return [
                {
                    "filename": row[0],
                    "username": row[1],
                    "size_bytes": row[2],
                    "uploaded_at": row[3],
                }
                for row in cursor.fetchall()
            ]

def _list_chroma_document_names(collection_name: str | None) -> list[str]:
    if not collection_name:
        return []

    try:
        collection = _get_chroma_client().get_collection(name=collection_name)
        results = collection.get(include=["metadatas"])
    except Exception as exc:
        log_error(
            logger,
            "document_options_chroma_read_failed",
            collection_name=collection_name,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return []

    document_names: set[str] = set()
    for metadata in results.get("metadatas") or []:
        if not metadata:
            continue
        document_name = str(metadata.get("document_name", "")).strip()
        if document_name:
            document_names.add(document_name)

    names = sorted(document_names, key=str.lower)
    log_info(
        logger,
        "document_options_chroma_read_completed",
        collection_name=collection_name,
        document_count=len(names),
    )
    return names

def _get_document_options(user: dict) -> list[dict]:
    options = [{"value": "", "label": "All policy documents"}]
    seen: set[tuple[str, str]] = set()

    def append_option(scope: str, filename: str, label_suffix: str) -> None:
        normalized_filename = (filename or "").strip()
        key = (scope, normalized_filename)
        if not normalized_filename or key in seen:
            return
        seen.add(key)
        options.append(
            {
                "value": f"{scope}::{normalized_filename}",
                "label": f"{normalized_filename} ({label_suffix})",
            }
        )

    for filename in _list_chroma_document_names(CHROMA_SHARED_COLLECTION_NAME):
        append_option("shared", filename, "Shared")

    for item in _list_uploaded_files("shared"):
        append_option("shared", item["filename"], "Shared")

    personal_collection_name = get_personal_collection_name(user.get("username", ""))
    for filename in _list_chroma_document_names(personal_collection_name):
        append_option("personal", filename, "Personal")

    for item in _list_uploaded_files("personal", user.get("username")):
        append_option("personal", item["filename"], "Personal")

    return options

def _parse_document_selection(selected_document: str | None) -> tuple[str | None, str | None]:
    if not selected_document:
        return None, None
    try:
        document_scope, document_name = selected_document.split("::", 1)
    except ValueError:
        return None, None
    document_scope = document_scope.strip().lower()
    document_name = document_name.strip()
    if document_scope not in {"shared", "personal"} or not document_name:
        return None, None
    return document_scope, document_name

def _add_uploaded_file(scope: str, filename: str, username: str | None, size_bytes: int | None) -> None:
    with _db_conn() as conn:
        with conn.cursor() as cursor:
            if scope == "personal" and username:
                cursor.execute(
                    "DELETE FROM uploaded_files WHERE scope = %s AND username = %s",
                    (scope, username),
                )
            cursor.execute(
                """
                INSERT INTO uploaded_files (filename, scope, username, size_bytes)
                VALUES (%s, %s, %s, %s)
                """,
                (filename, scope, username, size_bytes),
            )

def _delete_uploaded_file(scope: str, filename: str, username: str | None) -> None:
    with _db_conn() as conn:
        with conn.cursor() as cursor:
            if username:
                cursor.execute(
                    "DELETE FROM uploaded_files WHERE scope = %s AND filename = %s AND username = %s",
                    (scope, filename, username),
                )
            else:
                cursor.execute(
                    "DELETE FROM uploaded_files WHERE scope = %s AND filename = %s",
                    (scope, filename),
                )

def _get_user_activity(username: str, limit: int = 100) -> list[dict]:
    with _db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT event_type, resource_type, resource_id, metadata, ip, user_agent, created_at
                FROM audit_events
                WHERE username = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (username, limit),
            )
            rows = cursor.fetchall()
            return [
                {
                    "event_type": row[0],
                    "resource_type": row[1],
                    "resource_id": row[2],
                    "metadata": row[3] or {},
                    "ip": row[4],
                    "user_agent": row[5],
                    "created_at": row[6],
                }
                for row in rows
            ]

def _get_recent_queries(username: str, limit: int = 5) -> list[dict]:
    with _db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT metadata, created_at
                FROM audit_events
                WHERE username = %s AND event_type = 'QUERY'
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (username, limit),
            )
            rows = cursor.fetchall()
            return [
                {"metadata": row[0] or {}, "created_at": row[1]}
                for row in rows
            ]

def _save_query_history(
    username: str,
    role: str | None,
    scope: str,
    question: str,
    answer: str | None,
    justification: str | None,
    sources: list[dict] | None,
) -> None:
    with _db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO query_history (username, role, scope, question, answer, justification, sources)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    username,
                    role,
                    scope,
                    question,
                    answer,
                    justification,
                    psycopg2.extras.Json(sources or []),
                ),
            )

def _get_query_history(username: str, limit: int = 50) -> list[dict]:
    with _db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, scope, question, answer, justification, sources, created_at
                FROM query_history
                WHERE username = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (username, limit),
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "scope": row[1],
                    "question": row[2],
                    "answer": row[3],
                    "justification": row[4],
                    "sources": row[5] or [],
                    "created_at": row[6],
                }
                for row in rows
            ]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = _get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    if user.get("role") == "admin":
        return RedirectResponse(url="/admin", status_code=status.HTTP_302_FOUND)
    return RedirectResponse(url="/app", status_code=status.HTTP_302_FOUND)

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, user: dict = Depends(_require_admin)):
    csrf_token = _ensure_csrf_token(request)
    return templates.TemplateResponse(
        request=request,
        name="admin.html",
        context={
            "request": request,
            "user": user,
            "csrf_token": csrf_token,
        },
    )

@app.get("/admin/all", response_class=HTMLResponse)
async def admin_all(request: Request, user: dict = Depends(_require_admin)):
    csrf_token = _ensure_csrf_token(request)
    return templates.TemplateResponse(
        request=request,
        name="admin_all.html",
        context={
            "request": request,
            "user": user,
            "csrf_token": csrf_token,
            "document_options": _get_document_options(user),
        },
    )

@app.get("/admin/shared", response_class=HTMLResponse)
async def admin_shared(request: Request, user: dict = Depends(_require_admin)):
    shared_files = _list_uploaded_files("shared")
    csrf_token = _ensure_csrf_token(request)
    return templates.TemplateResponse(
        request=request,
        name="shared.html",
        context={"request": request, "user": user, "shared_files": shared_files, "csrf_token": csrf_token},
    )

@app.get("/app", response_class=HTMLResponse)
async def user_dashboard(request: Request, user: dict = Depends(_require_auth)):
    csrf_token = _ensure_csrf_token(request)
    return templates.TemplateResponse(
        request=request,
        name="user.html",
        context={
            "request": request,
            "user": user,
            "csrf_token": csrf_token,
        },
    )

@app.get("/app/all", response_class=HTMLResponse)
async def user_all(request: Request, user: dict = Depends(_require_auth)):
    csrf_token = _ensure_csrf_token(request)
    return templates.TemplateResponse(
        request=request,
        name="user_all.html",
        context={
            "request": request,
            "user": user,
            "csrf_token": csrf_token,
            "document_options": _get_document_options(user),
        },
    )

@app.get("/app/shared", response_class=HTMLResponse)
async def user_shared(request: Request, user: dict = Depends(_require_auth)):
    shared_files = _list_uploaded_files("shared")
    csrf_token = _ensure_csrf_token(request)
    return templates.TemplateResponse(
        request=request,
        name="shared.html",
        context={"request": request, "user": user, "shared_files": shared_files, "csrf_token": csrf_token},
    )

@app.get("/activity", response_class=HTMLResponse)
async def activity_page(request: Request, user: dict = Depends(_require_auth)):
    events = _get_user_activity(user["username"])
    csrf_token = _ensure_csrf_token(request)
    return templates.TemplateResponse(
        request=request,
        name="activity.html",
        context={"request": request, "user": user, "events": events, "csrf_token": csrf_token},
    )

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = _get_current_user(request)
    if user:
        if user.get("role") == "admin":
            return RedirectResponse(url="/admin", status_code=status.HTTP_302_FOUND)
        return RedirectResponse(url="/app", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"request": request, "csrf_token": _ensure_csrf_token(request)},
    )

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    csrf_token: str | None = Form(None),
):
    _require_csrf(request, csrf_token)
    username = username.strip()
    record = _get_user_record(username)
    if record and record.get("locked_until"):
        if record["locked_until"] > datetime.now(timezone.utc):
            _record_audit_event(
                username,
                record.get("role"),
                "LOGIN_LOCKED",
                request,
                metadata={"locked_until": record["locked_until"].isoformat()},
            )
            return templates.TemplateResponse(
                request=request,
                name="login.html",
                context={"request": request, "error": "Account is temporarily locked. Try again later."},
                status_code=status.HTTP_403_FORBIDDEN,
            )

    user = _verify_user(username, password)
    if not user:
        if record:
            failed_attempts = (record.get("failed_attempts") or 0) + 1
            locked_until = None
            if failed_attempts >= MAX_FAILED_ATTEMPTS:
                locked_until = datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)
                failed_attempts = 0
            with _db_conn() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE users
                        SET failed_attempts = %s, locked_until = %s
                        WHERE username = %s
                        """,
                        (failed_attempts, locked_until, username),
                    )
            _record_audit_event(
                username,
                record.get("role"),
                "LOGIN_FAIL",
                request,
            )
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"request": request, "error": "Invalid username or password."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    with _db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE users
                SET failed_attempts = 0, locked_until = NULL, last_login_at = NOW()
                WHERE username = %s
                """,
                (username,),
            )
    request.session.clear()
    request.session["user"] = user
    _record_audit_event(user["username"], user["role"], "LOGIN_SUCCESS", request)
    if user.get("role") == "admin":
        return RedirectResponse(url="/admin", status_code=status.HTTP_302_FOUND)
    return RedirectResponse(url="/app", status_code=status.HTTP_302_FOUND)

@app.post("/register")
async def register(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    csrf_token: str | None = Form(None),
):
    _require_csrf(request, csrf_token)
    accept = request.headers.get("accept", "")
    is_fetch = request.headers.get("x-requested-with") == "fetch" or "application/json" in accept
    username = username.strip()
    if not username or not password:
        message = "Username and password are required."
        if is_fetch:
            return {"ok": False, "message": message, "error_code": "missing_fields"}
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"request": request, "register_error": message},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    ok, message = _create_user(username, password)
    if not ok:
        error_code = "password_invalid" if message.startswith("Password") else "register_failed"
        if is_fetch:
            return {"ok": False, "message": message, "error_code": error_code}
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"request": request, "register_error": message},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    _record_audit_event(username, "user", "REGISTER", request)
    if is_fetch:
        return {"ok": True, "message": message}
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"request": request, "register_success": message},
        status_code=status.HTTP_201_CREATED,
    )

@app.get("/logout")
async def logout(request: Request):
    user = _get_current_user(request)
    if user:
        _record_audit_event(user.get("username"), user.get("role"), "LOGOUT", request)
    request.session.clear()
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@app.post("/upload-policy")
async def upload_policy(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(_require_admin),
    csrf_token: str | None = Form(None),
):
    _require_csrf(request, csrf_token)
    try:
        # Save the uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the new policy
        pipeline = IngestionPipeline(
            dataset_dir=str(UPLOAD_DIR),
            collection_name=CHROMA_SHARED_COLLECTION_NAME,
            file_paths=[str(file_path)],
        )
        pipeline.run()
        log_info(
            logger,
            "shared_upload_completed",
            username=user.get("username"),
            filename=file.filename,
            collection_name=CHROMA_SHARED_COLLECTION_NAME,
        )

        _add_uploaded_file("shared", file.filename, user.get("username"), file_path.stat().st_size)
        _record_audit_event(
            user.get("username"),
            user.get("role"),
            "UPLOAD_SHARED",
            request,
            resource_type="pdf",
            resource_id=file.filename,
            metadata={"filename": file.filename},
        )
        return {"message": f"Successfully processed policy: {file.filename}"}
    except Exception as e:
        log_error(
            logger,
            "shared_upload_failed",
            username=user.get("username"),
            filename=file.filename if file else None,
            error_type=type(e).__name__,
            error=str(e),
        )
        logger.exception("shared_upload_exception")
        return {"error": str(e)}

@app.post("/upload-personal")
async def upload_personal(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(_require_auth),
    csrf_token: str | None = Form(None),
):
    _require_csrf(request, csrf_token)
    try:
        user_dir = PERSONAL_UPLOAD_DIR / user["username"]
        user_dir.mkdir(parents=True, exist_ok=True)
        # Allow only one personal file per user by clearing existing uploads.
        for existing in user_dir.glob("*"):
            if existing.is_file():
                existing.unlink()

        file_path = user_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pipeline = IngestionPipeline(
            dataset_dir=str(user_dir),
            collection_name=get_personal_collection_name(user["username"]),
            file_paths=[str(file_path)],
        )
        pipeline.run()
        log_info(
            logger,
            "personal_upload_completed",
            username=user.get("username"),
            filename=file.filename,
            collection_name=get_personal_collection_name(user["username"]),
        )

        _add_uploaded_file("personal", file.filename, user.get("username"), file_path.stat().st_size)
        _record_audit_event(
            user.get("username"),
            user.get("role"),
            "UPLOAD_PERSONAL",
            request,
            resource_type="pdf",
            resource_id=file.filename,
            metadata={"filename": file.filename},
        )
        return {"message": f"Personal policy uploaded: {file.filename}"}
    except Exception as e:
        log_error(
            logger,
            "personal_upload_failed",
            username=user.get("username"),
            filename=file.filename if file else None,
            error_type=type(e).__name__,
            error=str(e),
        )
        logger.exception("personal_upload_exception")
        return {"error": str(e)}

@app.get("/files/shared/{filename}")
async def get_shared_file(filename: str, user: dict = Depends(_require_auth)):
    file_path = _safe_file_path(UPLOAD_DIR, filename)
    return FileResponse(file_path, media_type="application/pdf", filename=filename)

@app.get("/files/personal/{filename}")
async def get_personal_file(filename: str, user: dict = Depends(_require_auth)):
    user_dir = PERSONAL_UPLOAD_DIR / user["username"]
    file_path = _safe_file_path(user_dir, filename)
    return FileResponse(file_path, media_type="application/pdf", filename=filename)

@app.post("/delete/shared/{filename}")
async def delete_shared_file(
    request: Request,
    filename: str,
    user: dict = Depends(_require_admin),
    csrf_token: str | None = Form(None),
):
    _require_csrf(request, csrf_token)
    file_path = _safe_file_path(UPLOAD_DIR, filename)
    file_path.unlink()
    _delete_uploaded_file("shared", filename, None)
    _record_audit_event(
        user.get("username"),
        user.get("role"),
        "DELETE_SHARED",
        request,
        resource_type="pdf",
        resource_id=filename,
        metadata={"filename": filename},
    )
    return {"message": f"Deleted {filename}"}

@app.post("/delete/personal/{filename}")
async def delete_personal_file(
    request: Request,
    filename: str,
    user: dict = Depends(_require_auth),
    csrf_token: str | None = Form(None),
):
    _require_csrf(request, csrf_token)
    user_dir = PERSONAL_UPLOAD_DIR / user["username"]
    file_path = _safe_file_path(user_dir, filename)
    file_path.unlink()
    _delete_uploaded_file("personal", filename, user.get("username"))
    _record_audit_event(
        user.get("username"),
        user.get("role"),
        "DELETE_PERSONAL",
        request,
        resource_type="pdf",
        resource_id=filename,
        metadata={"filename": filename},
    )
    return {"message": f"Deleted {filename}"}

_tavily_service: TavilyService | None = None

def _get_tavily_service() -> TavilyService:
    global _tavily_service
    if _tavily_service is None:
        client = TavilySearchClient()
        _tavily_service = TavilyService(client)
    return _tavily_service


def _build_retrieval_debug(chunks: list[dict] | None) -> dict:
    chunks = chunks or []
    source_counts = {"semantic": 0, "keyword": 0, "both": 0}
    top_chunks: list[dict] = []

    for chunk in chunks[:5]:
        matched_by = chunk.get("matched_by", []) or []
        matched_set = set(matched_by)
        if matched_set == {"semantic"}:
            source_counts["semantic"] += 1
        elif matched_set == {"keyword"}:
            source_counts["keyword"] += 1
        elif matched_set:
            source_counts["both"] += 1

        metadata = chunk.get("metadata", {}) or {}
        top_chunks.append(
            {
                "document": metadata.get("document_name", "unknown_document"),
                "section": metadata.get("section_heading", "General"),
                "matched_by": sorted(matched_set),
                "hybrid_score": chunk.get("hybrid_score"),
                "semantic_score": chunk.get("semantic_score"),
                "keyword_score": chunk.get("keyword_score"),
            }
        )

    return {
        "top_chunk_count": len(chunks[:5]),
        "source_counts": source_counts,
        "top_chunks": top_chunks,
    }


def _chunk_log_summary(chunks: list[dict] | None, limit: int = 5) -> list[dict]:
    summary: list[dict] = []
    for chunk in (chunks or [])[:limit]:
        metadata = chunk.get("metadata", {}) or {}
        summary.append(
            {
                "document": metadata.get("document_name", "unknown_document"),
                "section": metadata.get("section_heading", "General"),
                "distance": chunk.get("distance"),
                "hybrid_score": chunk.get("hybrid_score"),
                "semantic_score": chunk.get("semantic_score"),
                "keyword_score": chunk.get("keyword_score"),
                "matched_by": chunk.get("matched_by", []),
                "text_preview": str(chunk.get("text", ""))[:220],
            }
        )
    return summary


@app.post("/query")
async def query(
    request: Request,
    query: str = Form(...),
    scope: str = Form("shared"),
    selected_document: str = Form(""),
    user: dict = Depends(_require_auth),
    csrf_token: str | None = Form(None),
):
    _require_csrf(request, csrf_token)
    query_log_token = None
    try:
        selected_scope, document_filter = _parse_document_selection(selected_document)
        effective_scope = scope
        collection_name = None
        if selected_scope == "shared":
            effective_scope = "shared"
            collection_name = CHROMA_SHARED_COLLECTION_NAME
        elif selected_scope == "personal":
            effective_scope = "personal"
            collection_name = get_personal_collection_name(user["username"])

        query_log_token = start_query_log(
            current_transaction_id(),
            logger=logger,
            username=user.get("username"),
            scope=effective_scope,
            selected_document=document_filter,
            collection_name=collection_name,
            generated=query,
        )

        _record_audit_event(
            user.get("username"),
            user.get("role"),
            "QUERY",
            request,
            metadata={
                "scope": effective_scope,
                "query_length": len(query),
                "selected_document": document_filter,
            },
        )
        log_info(
            logger,
            "query_processing_started",
            username=user.get("username"),
            scope=effective_scope,
            selected_document=document_filter,
            collection_name=collection_name,
        )
        log_query_step(
            logger,
            "query_pipeline_entered",
            generated=query,
            username=user.get("username"),
            scope=effective_scope,
            selected_document=document_filter,
            collection_name=collection_name,
        )
        
        if USE_ORCHESTRATION:
            # Use LangGraph orchestration for intelligent routing
            orchestrator = _get_orchestrator()
            result = await orchestrator.process_query(
                query=query,
                scope=effective_scope,
                username=user.get("username"),
                collection_name=collection_name,
                document_filter=document_filter,
            )
            
            response_text = result.get("response", "")
            justification_text = result.get("justification")
            sources = result.get("sources", [])
            route_taken = result.get("route_taken", "unknown")
            retrieval_debug = result.get("retrieval_debug")
            
            log_info(
                logger,
                "query_orchestration_completed",
                username=user.get("username"),
                route_taken=route_taken,
                source_count=len(sources),
            )
            log_query_step(
                logger,
                "query_response_ready",
                generated=response_text,
                username=user.get("username"),
                route_taken=route_taken,
                source_count=len(sources),
                justification=justification_text,
            )
            
            _save_query_history(
                user.get("username"),
                user.get("role"),
                effective_scope,
                query,
                response_text,
                justification_text,
                sources,
            )

            return {
                "response": response_text,
                "justification": justification_text,
                "sources": sources,
                "route_taken": route_taken,  # Include route info in response
                "retrieval_debug": retrieval_debug,
            }
        else:
            # Legacy path: Direct RAG processing without orchestration
            rewritten_query = query
            rewriter = _get_rewriter()
            rewritten_query = await rewriter.rewrite_query(query) or query
            log_query_step(
                logger,
                "query_rewrite",
                generated=rewritten_query,
                original_query=query,
            )

            retriever = _get_retriever()
            chunks: list[dict] = []
            if effective_scope == "shared":
                chunks = retriever.retrive_Chunks(
                    rewritten_query,
                    collection_name=CHROMA_SHARED_COLLECTION_NAME,
                    document_filter=document_filter,
                )
            elif effective_scope == "personal":
                chunks = retriever.retrive_Chunks(
                    rewritten_query,
                    collection_name=get_personal_collection_name(user["username"]),
                    document_filter=document_filter,
                )
            elif effective_scope == "combined":
                chunks = retriever.retrive_Chunks(
                    rewritten_query,
                    collection_name=CHROMA_SHARED_COLLECTION_NAME,
                    document_filter=document_filter,
                )
                chunks += retriever.retrive_Chunks(
                    rewritten_query,
                    collection_name=get_personal_collection_name(user["username"]),
                    document_filter=document_filter,
                )
            else:
                return {"error": f"Unknown scope: {scope}"}

            log_query_step(
                logger,
                "document_retrieval",
                generated=_chunk_log_summary(chunks),
                retrieved_count=len(chunks),
                rewritten_query=rewritten_query,
            )

            # if not chunks:
            #     if not chunks:
            #         tavily = _get_tavily_service()
            #         tavily_result = tavily.get_answer(query)

            #         _record_audit_event(
            #             user.get("username"),
            #             user.get("role"),
            #             "QUERY_EXTERNAL",
            #             request,
            #             metadata={"provider": "tavily"}
            #         )

            #         _save_query_history(
            #             user.get("username"),
            #             user.get("role"),
            #             "external",
            #             query,
            #             tavily_result["answer"],
            #             "Answer generated using Tavily web search",
            #             tavily_result["sources"],
            #         )

            #         return {
            #             "response": tavily_result["answer"],
            #             "sources": tavily_result["sources"],
            #             "route_taken": "tavily_fallback"
            #         }


            reranker = _get_reranker()
            answer_generator = _get_answer_generator()
            reranked = await reranker.rerank_chunks(rewritten_query, chunks, top_k=5)
            log_query_step(
                logger,
                "chunk_reranking",
                generated=_chunk_log_summary(reranked),
                reranked_count=len(reranked),
            )
            answer = await answer_generator.generate_answer(rewritten_query, reranked)
            response_text = answer.get("answer", "")
            justification_text = answer.get("justification")
            sources = answer.get("source_chunks", [])
            retrieval_debug = _build_retrieval_debug(reranked)
            log_query_step(
                logger,
                "answer_generation",
                generated=response_text,
                justification=justification_text,
                source_count=len(sources),
            )
            log_info(
                logger,
                "query_legacy_rag_completed",
                username=user.get("username"),
                scope=effective_scope,
                chunk_count=len(chunks),
                reranked_count=len(reranked),
                source_count=len(sources),
            )
            log_query_step(
                logger,
                "query_response_ready",
                generated=response_text,
                username=user.get("username"),
                route_taken="legacy_rag",
                source_count=len(sources),
                justification=justification_text,
            )

            _save_query_history(
                user.get("username"),
                user.get("role"),
                effective_scope,
                query,
                response_text,
                justification_text,
                sources,
            )

            return {
                "response": response_text,
                "justification": justification_text,
                "sources": sources,
                "retrieval_debug": retrieval_debug,
            }
    except Exception as e:
        log_query_error(
            logger,
            "query_processing_failed",
            generated=str(e),
            username=user.get("username"),
            scope=scope,
            error_type=type(e).__name__,
        )
        log_error(
            logger,
            "query_processing_failed",
            username=user.get("username"),
            scope=scope,
            error_type=type(e).__name__,
            error=str(e),
        )
        logger.exception("query_processing_exception")
        return {"error": str(e)}
    finally:
        if query_log_token is not None:
            reset_query_id(query_log_token)

@app.get("/history")
async def history(
    user: dict = Depends(_require_auth),
    limit: int = 50,
):
    return {"items": _get_query_history(user.get("username"), limit=limit)}


