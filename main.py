import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from data.load import *
from rag.embed import *
from pprint import pprint 

def load_llm():
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set your OPENAI_API_KEY as an environment variable.")
        exit(1)

    print("Initializing ML Debugging Assistant...")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # gpt-4o-mini is fast and cheap

if __name__ == '__main__':
    # llm = load_llm()
    
    documents = load_documents()
    ## Reference for RAG
    pprint(documents)
    
    vector_embed = create_vector_store(documents)
    pprint(vector_embed.shape, vector_embed.type)
    
    print("FAISS Vector Database loaded successfully!\n")
    
    