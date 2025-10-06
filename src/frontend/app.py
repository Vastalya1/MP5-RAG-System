from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import shutil
import os
from pathlib import Path
from dotenv import load_dotenv

# Import all RAG pipeline components
from src.ingestion.ingestionPipeline import IngestionPipeline
from src.queryRewriter.rewriting import QueryRewriter
from src.retriever.retrival import retrivalModel
from src.retriever.reranking import ChunkReranker
from src.output.answerGeneration import AnswerGenerator


# Initialize all components with API keys
MISTRAL_API_KEY = "IQEYxF0lNL2msXCKTg1ryDz9v3tSiQ59"  # Replace with your Mistral API key
GEMINI_API_KEY = "AIzaSyBLJ_eYaCBQ6TY4RUGf_gelHyU1H4pPw1g"  # Replace with your Gemini API key

# Initialize components
# query_rewriter = QueryRewriter(GEMINI_API_KEY)
query_rewriter = QueryRewriter(MISTRAL_API_KEY)  # Now using Mistral API
retriever = retrivalModel()
reranker = ChunkReranker(GEMINI_API_KEY)
answer_generator = AnswerGenerator(GEMINI_API_KEY)

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

app = FastAPI()

# Setup template and static directories with absolute paths
templates = Jinja2Templates(directory=str(BASE_DIR / "src" / "frontend" / "templates"))
app.mount("/static", 
          StaticFiles(directory=str(BASE_DIR / "src" / "frontend" / "static")), 
          name="static")

# Ensure upload directory exists
UPLOAD_DIR = BASE_DIR / "dataset" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/upload-policy")
async def upload_policy(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the new policy
        pipeline = IngestionPipeline(
            dataset_dir=str(UPLOAD_DIR),
            collection_name="policy_documents"
        )
        pipeline.run()
        
        return {"message": f"Successfully processed policy: {file.filename}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/query")
async def query(query: str = Form(...)):
    try:
        print(f"Processing query: {query}")
        print(f"Using API key: {GEMINI_API_KEY[:10]}...")  # Print first 10 chars of API key for verification
        
        # Step 1: Rewrite the query using medical insurance terminology
        rewritten_query = await query_rewriter.rewrite_query(query)
        if rewritten_query is None:
            print("Query rewriting failed")
            return {"error": "Failed to rewrite query. Please check server logs for details."}
        print(f"✨ Successfully rewrote query: {rewritten_query}")

        # Step 2: Retrieve relevant chunks
        initial_chunks = retriever.retrive_Chunks(rewritten_query)
        if not initial_chunks:
            return {"error": "No relevant information found in the database."}
        print(f"📄 Retrieved {len(initial_chunks)} initial chunks")

        # Step 3: Rerank chunks
        reranked_chunks = await reranker.rerank_chunks(rewritten_query, initial_chunks)
        if not reranked_chunks:
            return {"error": "Failed to rerank chunks."}
        print(f"🔄 Reranked chunks successfully")

        # Step 4: Generate answer
        answer_response = await answer_generator.generate_answer(rewritten_query, reranked_chunks)
        
        # Step 5: Return formatted response
        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "answer": answer_response["answer"],
            "justification": answer_response["justification"],
            "sources": answer_response["source_chunks"]
        }
    except Exception as e:
        print(f"Error in query endpoint: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"An error occurred: {str(e)}"}