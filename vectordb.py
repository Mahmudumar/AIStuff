import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

local_embedding_model = ONNXMiniLM_L6_V2(
    download_dir="./models/chroma_model"
)

# Initialize a persistent client to save data to a local folder
client = chromadb.PersistentClient(path="./my_chroma_db")

# Create a new collection (or get it if it already exists)
collection = client.get_or_create_collection(name="my_documents")
collection.add(
    documents=[
        "ChromaDB is a lightweight vector database for AI.",
        "Python is a popular programming language for data science.",
        "A database is an organized collection of data."
    ],
    metadatas=[
        {"category": "tech"}, 
        {"category": "coding"}, 
        {"category": "tech"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

results = collection.query(
    query_texts=["What tool should I use for AI data stores?"],
    n_results=1 # Tells ChromaDB to return the single closest match
)

print(results["documents"])
# Output: [['ChromaDB is a lightweight vector database for AI.']]
