import uvicorn
from shared.logging_utils import configure_logging, get_logger


configure_logging()
logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("server_starting | host=0.0.0.0 port=8000 reload=false")
    uvicorn.run("frontend.app:app", host="0.0.0.0", port=8000, reload=False)
