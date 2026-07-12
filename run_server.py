import sys
import os
from pathlib import Path

# Add the project root directory to Python path
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))

import uvicorn
from src.shared.logging_utils import configure_logging, get_logger


configure_logging()
logger = get_logger(__name__)

if __name__ == "__main__":
    reload_enabled = os.getenv("UVICORN_RELOAD", "false").lower() in {"1", "true", "yes"}
    logger.info("server_starting | host=127.0.0.1 port=8000 reload=%s", reload_enabled)
    uvicorn.run("src.frontend.app:app", host="127.0.0.1", port=8000, reload=reload_enabled)
