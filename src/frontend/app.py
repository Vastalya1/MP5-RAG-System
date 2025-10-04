from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import shutil
import os
from pathlib import Path
from dotenv import load_dotenv

# Import your ingestion pipeline and other necessary components
from src.ingestion.ingestionPipeline import IngestionPipeline
from src.queryRewriter.rewriting import QueryRewriter


# Initialize the query rewriter with API key
GEMINI_API_KEY = "AIzaSyBLJ_eYaCBQ6TY4RUGf_gelHyU1H4pPw1g"  # Replace this with your actual Gemini API key
query_rewriter = QueryRewriter(GEMINI_API_KEY)

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
        
        # Rewrite the query using medical insurance terminology
        rewritten_query = await query_rewriter.rewrite_query(query)
        
        if rewritten_query is None:
            print("Query rewriting failed")
            return {"error": "Failed to rewrite query. Please check server logs for details."}
            
        print(f"Successfully rewrote query: {rewritten_query}")
        # Here you'll implement the RAG logic with the rewritten query
        # For now, returning the rewritten query
        return {
            "response": f"Original query: {query}\nRewritten query: {rewritten_query}"
        }
    except Exception as e:
        print(f"Error in query endpoint: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"An error occurred: {str(e)}"}