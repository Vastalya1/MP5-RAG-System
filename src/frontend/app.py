from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
import shutil
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Import all RAG pipeline components
from src.ingestion.ingestionPipeline import IngestionPipeline
from src.queryRewriter.rewriting import QueryRewriter
from src.retriever.retrival import retrivalModel
from src.retriever.reranking_mistral import ChunkReranker  # Using Mistral version
from src.output.answerGeneration_mistral import AnswerGenerator  # Using Mistral version


# Initialize all components with API keys
MISTRAL_API_KEY = "IQEYxF0lNL2msXCKTg1ryDz9v3tSiQ59"  # Replace with your Mistral API key
GEMINI_API_KEY = "AIzaSyBLJ_eYaCBQ6TY4RUGf_gelHyU1H4pPw1g"  # Replace with your Gemini API key

# Store the latest result
latest_result = None

# Initialize components
query_rewriter = QueryRewriter(MISTRAL_API_KEY)  # Using Mistral API
retriever = retrivalModel()
reranker = ChunkReranker(MISTRAL_API_KEY)  # Using Mistral for reranking
answer_generator = AnswerGenerator(MISTRAL_API_KEY)  # Now using Mistral for answer generation

# Get the base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
print("Base directory set to: ", BASE_DIR)  # Debug log
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
    print("Rendering home page")  # Debug log
    try:
        response = templates.TemplateResponse(
            "index.html",
            {"request": request}
        )
        print("Template response created successfully")  # Debug log
        return response
    except Exception as e:
        print(f"Error rendering template: {str(e)}")  # Debug log
        raise

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
async def query(request: Request, query: str = Form(...)):
    global latest_result
    try:
        print(f"Processing query: {query}")
        # Check if the request is from browser (HTML) or API call
        is_api_request = request.headers.get("accept") == "application/json"
        
        # Step 1: Rewrite the query using medical insurance terminology
        rewritten_query = await query_rewriter.rewrite_query(query)
        if rewritten_query is None:
            print("Query rewriting failed")
            error_response = {"error": "Failed to rewrite query. Please check server logs for details."}
            return error_response if is_api_request else templates.TemplateResponse(
                "index.html", {"request": request, "error": error_response["error"]}
            )
        print(f"✨ Successfully rewrote query: {rewritten_query}")

        # Step 2: Retrieve relevant chunks
        initial_chunks = retriever.retrive_Chunks(rewritten_query)
        if not initial_chunks:
            error_response = {"error": "No relevant information found in the database."}
            return error_response if is_api_request else templates.TemplateResponse(
                "index.html", {"request": request, "error": error_response["error"]}
            )
        print(f"📄 Retrieved {len(initial_chunks)} initial chunks")

        # Step 3: Rerank chunks
        reranked_chunks = await reranker.rerank_chunks(rewritten_query, initial_chunks)
        if not reranked_chunks:
            return {"error": "Failed to rerank chunks."}
        print(f"🔄 Reranked chunks successfully")

        # Step 4: Generate answer
        answer_response = await answer_generator.generate_answer(rewritten_query, reranked_chunks)
        
        # Store the result
        latest_result = {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "answer": answer_response["answer"],
            "justification": answer_response["justification"],
            "sources": answer_response["source_chunks"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Step 5: Return response based on request type
        if is_api_request:
            return latest_result
        else:
            return RedirectResponse(url="/result", status_code=303)
        
    except Exception as e:
        print(f"Error in query endpoint: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        error_response = {"error": f"An error occurred: {str(e)}"}
        return error_response if is_api_request else templates.TemplateResponse(
            "index.html", {"request": request, "error": error_response["error"]}
        )

@app.get("/result")
async def get_latest_result(request: Request):
    if latest_result is None:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "No query has been processed yet"}
        )
    
    # Check if the request accepts HTML or JSON
    is_api_request = request.headers.get("accept") == "application/json"
    if is_api_request:
        return latest_result
    
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "result": latest_result
        }
    )