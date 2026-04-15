import os
from pathlib import Path

from dotenv import load_dotenv


_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)


def get_shared_collection_name() -> str:
    return os.getenv("CHROMA_SHARED_COLLECTION_NAME", "dataset")


def get_personal_collection_prefix() -> str:
    return os.getenv("CHROMA_PERSONAL_COLLECTION_PREFIX", "user_")


def get_personal_collection_suffix() -> str:
    return os.getenv("CHROMA_PERSONAL_COLLECTION_SUFFIX", "_documents")


def get_personal_collection_name(username: str) -> str:
    return f"{get_personal_collection_prefix()}{username}{get_personal_collection_suffix()}"
