import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv


env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("CHROMA_CLOUD_API_KEY") or os.getenv("CHROMA_API_KEY")
if not api_key:
    raise ValueError(
        f"Missing Chroma API key. Set CHROMA_CLOUD_API_KEY or CHROMA_API_KEY in {env_path}."
    )

client = chromadb.CloudClient(
    api_key=api_key,
    tenant="a92961b0-ea65-4a82-a7ad-321a4baaaa60",
    database="Major-Project",
)


# collection = client.get_collection("temp_dataset")
# results = collection.get(limit=5)
# for i, doc in enumerate(results["documents"]):
#     print(f"Chunk {i + 1}: {doc}")


collection_name = "temp_dataset"

client.delete_collection(collection_name)
client.create_collection(collection_name)

collections = client.list_collections()
for collection in collections:
    print(collection.name)
