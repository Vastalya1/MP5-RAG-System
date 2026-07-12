# MP5-RAG-System

This project is an AI-powered assistant for medical insurance policy documents. It uses a Retrieval-Augmented Generation (RAG) pipeline to answer user questions with grounded policy context instead of relying on free-form generation alone.

Framework: FastAPI backend + Jinja2 UI

Embedding Model: Sentence-BERT (`all-MiniLM-L6-v2`)

Vector Database: ChromaDB Cloud with cosine similarity search

Active LLM Stack: OpenAI-backed modules (`gpt-4o-mini`) for query rewriting, routing, reranking, answer generation, and table-to-sentence enrichment

Note: The repository still contains older Mistral-based variants for evaluation and experimentation, but the current FastAPI app uses the OpenAI-backed `*_Chatgpt.py` modules.

## Modules Implemented

- Document ingestion (PDF chunking + embeddings)
- Query rewriting
- Hybrid retrieval
- Metadata + LLM reranking
- Answer generation
- LangGraph-based routing and orchestration
- Role-based authentication (admin/user)
- Shared and personal policy uploads
- Activity and query history logging
- Tavily-based web search fallback
- Voice input in chat UI

## Retrieval Pipeline

The current app path implements hybrid retrieval:

- Dense retrieval: semantic vector search using ChromaDB Cloud
- Sparse retrieval: BM25 keyword search using `rank-bm25`
- Fusion: Reciprocal Rank Fusion (RRF)
- Reranking: metadata-aware reranking followed by LLM reranking

This improves recall for paraphrased queries while still capturing exact policy terms such as clause names, waiting periods, ICU limits, room-rent caps, and similar insurance-specific wording.

## Setup

Create a fresh Python virtual environment from the project root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py -3.13` is not available on your machine, use your installed Python 3.13 executable instead.

## Environment

The app loads configuration from `.env` in the project root.

If `.env` does not exist yet, create it and add the required values below.

Required values:

```env
DATABASE_URL=your_postgresql_connection_string
SESSION_SECRET=change_me
OPENAI_API_KEY=your_key_here
CHROMA_CLOUD_API_KEY=your_chroma_cloud_key
```

Common optional values:

```env
USE_ORCHESTRATION=true
TAVILY_API_KEY=your_tavily_key
DISABLE_CSRF=false
SESSION_MAX_AGE=1800
SESSION_HTTPS_ONLY=false
MAX_FAILED_ATTEMPTS=5
LOCKOUT_MINUTES=15
PASSWORD_MIN_LENGTH=8
CHROMA_CLOUD_TENANT=your_tenant_id
CHROMA_CLOUD_DATABASE=your_database_name
CHROMA_SHARED_COLLECTION_NAME=dataset
CHROMA_PERSONAL_COLLECTION_PREFIX=user_
CHROMA_PERSONAL_COLLECTION_SUFFIX=_documents
UVICORN_RELOAD=false
```

Notes:

- `OPENAI_API_KEY` is required for the active query pipeline.
- `TAVILY_API_KEY` is only needed if you want Tavily web-search fallback enabled.
- The active ingestion and retrieval path uses Chroma Cloud, not local `CHROMA_PERSIST_DIR` storage.

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

These accounts are seeded on startup:

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
- Activity log for uploads, deletions, logins, and queries
- Retrieval debug summary showing whether top chunks came from semantic retrieval, keyword retrieval, or both
- Orchestrated routing between direct LLM, RAG, and Tavily web-search fallback paths

## Storage Paths

- Shared uploads: `dataset/uploads`
- Personal uploads: `dataset/user_uploads/<username>`
- Vector collections: Chroma Cloud shared collection plus per-user personal collections

## Dependencies Added For This Branch

- `rank-bm25` for sparse BM25 retrieval
- `tavily-python` for Tavily web fallback integration
- `langgraph` and `langchain-core` for orchestration

## Notes

- PostgreSQL must be reachable through `DATABASE_URL`.
- The app initializes database tables on startup.
- Hybrid retrieval is active when logs show semantic retrieval, keyword retrieval, and hybrid fusion steps for a query.
- The active app imports the OpenAI-backed modules under `src/queryRewriter`, `src/retriever`, `src/output`, `src/orchestration`, and `src/ingestion`.
