import sys
import os
from pathlib import Path

# Add the project root directory to Python path
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

import uvicorn

if __name__ == "__main__":
    reload_enabled = os.getenv("UVICORN_RELOAD", "false").lower() in {"1", "true", "yes"}
    uvicorn.run("src.frontend.app:app", host="127.0.0.1", port=8000, reload=reload_enabled)
