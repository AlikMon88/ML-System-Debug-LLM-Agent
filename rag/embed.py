def create_vector_store(documents):
    # Using OpenAI's optimized embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store