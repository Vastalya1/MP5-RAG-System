import chromadb

# Connect to your persistent Chroma DB
client = chromadb.PersistentClient(path=r"D:\_official_\_MIT ADT_\_SEMESTER 7_\MP5\MP5-RAG-System\chromadb")

# Replace 'my_collection' with the name of your collection
collection = client.get_collection("policy_documents")

# Get the first 5 documents/chunks
results = collection.get(limit=5)

# Print them
for i, doc in enumerate(results['documents']):
    print(f"Chunk {i+1}: {doc}")
