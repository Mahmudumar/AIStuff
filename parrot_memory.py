import chromadb
import ollama

# 1. Initialize ChromaDB Client
# Using a local persistent database to keep your data saved
chroma_client = chromadb.PersistentClient(path="./phi3_memory")
collection = chroma_client.get_or_create_collection(name="knowledge_base")

# 2. Define Sample Documents to Feed Your Local Database
documents_to_store = []
with open("doc1.txt", "r") as r:
    for line in r.readlines():
        if not line.strip() or line.startswith("<<>>"):
            continue
        documents_to_store.append(line.strip())
    # documents_to_store.append(r.read())
    
# document_ids = ["doc1", "doc2", "doc3","doc4", "doc5"]
document_ids = [f"doc{x+1}" for x in range(len(documents_to_store))]

# print(documents_to_store, document_ids)

# 3. Generate Embeddings via Ollama and Store Them in ChromaDB
# We loop through documents, calculate vectors locally, and store them
for doc, doc_id in zip(documents_to_store, document_ids):
    # Get vector embedding from Ollama
    response = ollama.embeddings(model="nomic-embed-text", prompt=doc)
    embedding = response["embedding"]
    
    # Add text and vector directly to ChromaDB
    collection.add(
        embeddings=[embedding],
        documents=[doc],
        ids=[doc_id]
    )

print("✅ Data successfully embedded and saved to ChromaDB!")

while True:
    # 4. User Asks a Question
    user_query = input('\n\nAsk me any question: ')

    # 5. Convert User Question to Vector & Query ChromaDB
    query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=user_query)["embedding"]

    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    # Extract the most relevant piece of text found
    retrieved_context = search_results["documents"][0] # type: ignore
    print(f"\n🔍 ChromaDB Found Context: '{retrieved_context}'")

    # 6. Build the Augmented Prompt and Query Phi-3 Mini
    # We blend the retrieved fact directly into the instructions for Phi-3
    system_prompt = f"""
    You are a helpful assistant named Parrot. Answer the user's question using ONLY the provided context.
    Context: {retrieved_context}
    """
    

    phi3_response = ollama.generate(
        model="phi3:mini",
        system=system_prompt,
        prompt=user_query,
        stream=True,
        keep_alive=5000.5
    )

    print("\n🤖 Phi-3 Mini Response:")
    for token in phi3_response:
        print(token.response, end="", flush=True)