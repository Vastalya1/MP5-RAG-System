from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import shutil
import os
from pathlib import Path

# Import your ingestion pipeline and other necessary components
from ..ingestion.ingestionPipeline import IngestionPipeline

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
        # Here you'll implement the query processing logic
        # For now, returning a placeholder
        return {
            "response": f"Your query: {query}\nThis is a placeholder response. Implement your RAG logic here."
        }
    except Exception as e:
        return {"error": str(e)}