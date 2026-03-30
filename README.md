# MP5-RAG-System

This project is an AI-powered assistant for medical insurance policy documents. It uses a Retrieval-Augmented Generation (RAG) pipeline to answer user questions with grounded policy context instead of relying on free-form generation alone.

**Framework:** FastAPI backend + Jinja2 UI

**Embedding Model:** Sentence-BERT (`all-MiniLM-L6-v2`)

**Vector Database:** ChromaDB with HNSW indexing

**LLM:** Mistral for query rewriting, reranking, routing, and answer generation

## Modules Implemented

- Document ingestion (PDF chunking + embeddings)
- Query rewriting
- Hybrid retrieval
- Metadata + LLM reranking
- Answer generation
- Role-based authentication (admin/user)
- Shared and personal policy uploads
- Activity and query history logging
- Tavily-based web search fallback
- Voice input in chat UI

## Retrieval Pipeline

The current branch implements hybrid retrieval:

- Dense retrieval: semantic vector search using ChromaDB
- Sparse retrieval: BM25 keyword search using `rank-bm25`
- Fusion: Reciprocal Rank Fusion (RRF)
- Reranking: metadata-aware reranking followed by LLM reranking

This improves recall for paraphrased queries while still capturing exact policy terms such as clause names, waiting periods, ICU limits, room-rent caps, and similar insurance-specific wording.

## Setup

Create a fresh Python 3.13 virtual environment from the project root:

```powershell
C:\Python313\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `C:\Python313\python.exe` is not available on your machine, use your installed Python 3.13 executable instead.

## Environment

The app loads configuration from `.env`.

If `.env` does not exist yet:

```powershell
Copy-Item .env.example .env
```

Required values:

```env
DATABASE_URL=your_database_url
SESSION_SECRET=change_me
MISTRAL_API_KEY=your_key_here
CHROMA_PERSIST_DIR=chromadb
CHROMA_CLOUD_API_KEY=your_chroma_cloud_key
```

Additional supported flags:

```env
DISABLE_CSRF=false
USE_ORCHESTRATION=true
TAVILY_API_KEY=your_tavily_key
SESSION_MAX_AGE=1800
SESSION_HTTPS_ONLY=false
MAX_FAILED_ATTEMPTS=5
LOCKOUT_MINUTES=15
PASSWORD_MIN_LENGTH=8
```

## Run

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python .\run_server.py
```

For development with auto-reload, use a free port such as `8001`:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.frontend.app:app --reload --host 127.0.0.1 --port 8001
```

Open:

- `http://127.0.0.1:8000` for `run_server.py`
- `http://127.0.0.1:8001` for the sample reload command above

## Default Admin Accounts

- Prachi / pk2026
- Sia / s2026
- Akshada / ak2026
- Vastalya / avs2026

## Chat Features

- Shared policy upload for admins
- Personal policy upload for users
- Text-based natural language queries
- Voice-to-text query input in supported browsers such as Chrome and Edge
- Query history with saved answers and sources
- Retrieval debug summary showing whether top chunks came from semantic retrieval, keyword retrieval, or both

## Storage Paths

- Shared uploads: `dataset/uploads`
- Personal uploads: `dataset/user_uploads/<username>`
- Chroma store: `chromadb` or `CHROMA_PERSIST_DIR`

## Dependencies Added For This Branch

- `rank-bm25` for sparse BM25 retrieval
- `tavily-python` for Tavily web fallback integration

## Notes

- PostgreSQL must be reachable through `DATABASE_URL`.
- The app initializes database tables on startup.
- Hybrid retrieval is active when the server logs show semantic retrieval, keyword retrieval, and hybrid fusion steps for a query.
