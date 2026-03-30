# MP5-RAG-System

The main objective of this project is to leverage current technology by developing an AI-powered system that is capable of answering accurately most user queries related to medical insurance policy documents and associated terminology. The proposed system integrates structured information retrieval with a reasoning-based generative model to ensure factual, contextually relevant responses. It is achieved with a Retrieval-Augmented Generation (RAG) framework, which grounds the model's outputs in verified policy data, minimizing hallucinations and enhancing interpretability for end users.

**Framework:** FastAPI backend + Jinja2 UI (RAG pipeline modules in Python).

**Embedding Model:** Sentence-BERT for vector creation.

**Vector Database:** ChromaDB with HNSW search.

**LLM:** Mistral for query rewrite and answer generation (env-configured).

## Modules Implemented

- Document ingestion (chunking + embeddings)
- Query rewrite
- Chunk retrieval
- Response generator
- Role-based auth (admin/user)
- Activity history logging

## Setup

This repo currently contains two local virtual environments:

- `.venv` (Python 3.8.10)
- `.venv313` (Python 3.13.7)

Activate whichever one you are using on your machine.

For the existing Python 3.13 environment:

```powershell
.\.venv313\Scripts\Activate.ps1
```

For the existing Python 3.8 environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

If you need to install packages into the active environment:

```powershell
pip install -r requirements.txt
```

## Environment

The app loads settings from `.env`.

If `.env` does not exist yet, create it from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Required values:

```env
DATABASE_URL=your_database_url
SESSION_SECRET=change_me
MISTRAL_API_KEY=your_key_here
CHROMA_PERSIST_DIR=chromadb
```

Additional flags currently supported by the app:

```env
DISABLE_CSRF=false
USE_ORCHESTRATION=true
TAVILY_API_KEY=your_tavily_key
CHROMA_CLOUD_API_KEY=your_chroma_cloud_key
```

## Run

From the project root, use either:

```powershell
python .\run_server.py
```

or:

```powershell
uvicorn src.frontend.app:app --reload
```

Open `http://127.0.0.1:8000`.

Notes:

- PostgreSQL must be reachable through `DATABASE_URL`.
- The app initializes database tables on startup.
- Voice input in the UI works in supported browsers such as Chrome or Edge.

## Default Admin Accounts

- Prachi / pk2026
- Sia / s2026
- Akshada / ak2026
- Vastalya / avs2026

## Storage Paths

- Shared uploads: `dataset/uploads`
- Personal uploads: `dataset/user_uploads/<username>`
- Chroma store: `chromadb` (or `CHROMA_PERSIST_DIR`)
