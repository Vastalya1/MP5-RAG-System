import chromadb

# Connect to your persistent Chroma DB
client = chromadb.PersistentClient(path=r"D:\_official_\_MIT ADT_\_SEMESTER 7_\MP5\MP5-RAG-System\chromadb")

# # Replace 'my_collection' with the name of your collection
# collection = client.get_collection("policy_documentSs")

# # Get the first 5 documents/chunks
# results = collection.get(limit=5)

# # Print them
# for i, doc in enumerate(results['documents']):
#     print(f"Chunk {i+1}: {doc}")



# collection_name = "policy_documents"

# # Delete the whole collection
# client.delete_collection(collection_name)

# # (Optional) Recreate it empty
# client.create_collection(collection_name)

# collections = client.list_collections()

# for c in collections:
#     print(c.name)



collection_name = "policy_documents"

# Delete the whole collection
client.delete_collection(collection_name)

# (Optional) Recreate it empty
client.create_collection(collection_name)

collections = client.list_collections()

for c in collections:
    print(c.name)
