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

python -m venv .venv


.\.venv\Scripts\Activate.ps1


pip install -r requirements.txt


Create `.env`  and set your keys:


Minimum env values:

DATABASE_URL=postgresql://mp5_user:your_password@localhost:5432/mp5_rag


SESSION_SECRET=change_me


MISTRAL_API_KEY=your_key_here


CHROMA_PERSIST_DIR=chromadb


## Run

python src\run_server.py


Open `http://127.0.0.1:8000`.

## Default Admin Accounts

- Prachi / pk2026
- Sia / s2026
- Akshada / ak2026
- Vastalya / avs2026

## Storage Paths

- Shared uploads: `dataset/uploads`
- Personal uploads: `dataset/user_uploads/<username>`
- Chroma store: `chromadb` (or `CHROMA_PERSIST_DIR`)
