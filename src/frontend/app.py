from fastapi import FastAPI, File, UploadFile, Form, Request, Depends, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from starlette.middleware.sessions import SessionMiddleware
import shutil
import os
from pathlib import Path
import hashlib
import hmac
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import secrets
from passlib.context import CryptContext
from dotenv import load_dotenv

from queryRewriter.rewriting import QueryRewriter
from retriever.retrival import retrivalModel
from retriever.reranking_mistral import ChunkReranker
from output.answerGeneration_mistral import AnswerGenerator

# Import your ingestion pipeline and other necessary components
from ingestion.ingestionPipeline import IngestionPipeline

# Load environment variables from .env (if present)
load_dotenv()

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

app = FastAPI()

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
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

_retriever_instance: retrivalModel | None = None
_rewriter_instance: QueryRewriter | None = None
_reranker_instance: ChunkReranker | None = None
_answer_generator_instance: AnswerGenerator | None = None

def _get_retriever() -> retrivalModel:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = retrivalModel()
    return _retriever_instance

def _get_rewriter() -> QueryRewriter:
    global _rewriter_instance
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY is required for query rewriting.")
    if _rewriter_instance is None:
        _rewriter_instance = QueryRewriter(MISTRAL_API_KEY)
    return _rewriter_instance

def _get_reranker() -> ChunkReranker:
    global _reranker_instance
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY is required for reranking.")
    if _reranker_instance is None:
        _reranker_instance = ChunkReranker(MISTRAL_API_KEY)
    return _reranker_instance

def _get_answer_generator() -> AnswerGenerator:
    global _answer_generator_instance
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY is required for answer generation.")
    if _answer_generator_instance is None:
        _answer_generator_instance = AnswerGenerator(MISTRAL_API_KEY)
    return _answer_generator_instance

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
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

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

def _require_csrf(request: Request, csrf_token: str) -> None:
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
    shared_files = _list_uploaded_files("shared")
    csrf_token = _ensure_csrf_token(request)
    recent_queries = _get_recent_queries(user["username"])
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "user": user,
            "shared_files": shared_files,
            "csrf_token": csrf_token,
            "recent_queries": recent_queries,
        }
    )

@app.get("/app", response_class=HTMLResponse)
async def user_dashboard(request: Request, user: dict = Depends(_require_auth)):
    shared_files = _list_uploaded_files("shared")
    personal_files = _list_uploaded_files("personal", user["username"])
    csrf_token = _ensure_csrf_token(request)
    recent_queries = _get_recent_queries(user["username"])
    return templates.TemplateResponse(
        "user.html",
        {
            "request": request,
            "user": user,
            "shared_files": shared_files,
            "personal_files": personal_files,
            "csrf_token": csrf_token,
            "recent_queries": recent_queries,
        }
    )

@app.get("/activity", response_class=HTMLResponse)
async def activity_page(request: Request, user: dict = Depends(_require_auth)):
    events = _get_user_activity(user["username"])
    csrf_token = _ensure_csrf_token(request)
    return templates.TemplateResponse(
        "activity.html",
        {"request": request, "user": user, "events": events, "csrf_token": csrf_token},
    )

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = _get_current_user(request)
    if user:
        if user.get("role") == "admin":
            return RedirectResponse(url="/admin", status_code=status.HTTP_302_FOUND)
        return RedirectResponse(url="/app", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "csrf_token": _ensure_csrf_token(request)}
    )

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    csrf_token: str = Form(...),
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
                "login.html",
                {"request": request, "error": "Account is temporarily locked. Try again later."},
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
            "login.html",
            {"request": request, "error": "Invalid username or password."},
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
    csrf_token: str = Form(...),
):
    _require_csrf(request, csrf_token)
    username = username.strip()
    if not username or not password:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "register_error": "Username and password are required."},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    ok, message = _create_user(username, password)
    if not ok:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "register_error": message},
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    _record_audit_event(username, "user", "REGISTER", request)
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "register_success": message},
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
    csrf_token: str = Form(...),
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
            collection_name="policy_documents",
            file_paths=[str(file_path)],
        )
        pipeline.run()

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
        return {"error": str(e)}

@app.post("/upload-personal")
async def upload_personal(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(_require_auth),
    csrf_token: str = Form(...),
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
            collection_name=f"user_{user['username']}_documents",
            file_paths=[str(file_path)],
        )
        pipeline.run()

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
    csrf_token: str = Form(...),
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
    csrf_token: str = Form(...),
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

@app.post("/query")
async def query(
    request: Request,
    query: str = Form(...),
    scope: str = Form("shared"),
    user: dict = Depends(_require_auth),
    csrf_token: str = Form(...),
):
    _require_csrf(request, csrf_token)
    try:
        _record_audit_event(
            user.get("username"),
            user.get("role"),
            "QUERY",
            request,
            metadata={"scope": scope, "query_length": len(query)},
        )
        rewritten_query = query
        rewriter = _get_rewriter()
        rewritten_query = await rewriter.rewrite_query(query) or query

        retriever = _get_retriever()
        chunks: list[dict] = []
        if scope == "shared":
            chunks = retriever.retrive_Chunks(rewritten_query, collection_name="policy_documents")
        elif scope == "personal":
            chunks = retriever.retrive_Chunks(
                rewritten_query,
                collection_name=f"user_{user['username']}_documents",
            )
        elif scope == "combined":
            chunks = retriever.retrive_Chunks(rewritten_query, collection_name="policy_documents")
            chunks += retriever.retrive_Chunks(
                rewritten_query,
                collection_name=f"user_{user['username']}_documents",
            )
        else:
            return {"error": f"Unknown scope: {scope}"}

        if not chunks:
            return {"response": "No relevant policy content found for this question."}

        reranker = _get_reranker()
        answer_generator = _get_answer_generator()
        reranked = await reranker.rerank_chunks(rewritten_query, chunks, top_k=5)
        answer = await answer_generator.generate_answer(rewritten_query, reranked)

        return {
            "response": answer.get("answer", ""),
            "justification": answer.get("justification"),
            "sources": answer.get("source_chunks", []),
        }
    except Exception as e:
        return {"error": str(e)}
