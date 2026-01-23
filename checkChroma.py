import chromadb

# Connect to your persistent Chroma DB
client = chromadb.PersistentClient(path=r"C:\Users\kunji\OneDrive\Pictures\Desktop\Major_Project\MP5-RAG-System\dataset")

# Reset the collection to empty
try:
    client.delete_collection("policy_documents")
    print("Deleted existing policy_documents collection.")
except Exception:
    print("No existing policy_documents collection to delete.")

client.get_or_create_collection("policy_documents")
print("Created empty policy_documents collection.")
