import sys
from pathlib import Path

# Add the project root directory to Python path
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.frontend.app:app", host="127.0.0.1", port=8000, reload=True)