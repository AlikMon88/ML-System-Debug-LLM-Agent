import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

def load_documents(file_path="data/incidents.json"):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    documents =[]
    for item in data:
        # Combine symptom, root cause, and resolution into the page content for context
        content = f"Symptom: {item['symptom']}\nRoot Cause: {item['root_cause']}\nResolution: {item['resolution']}"
        metadata = {"incident_id": item["incident_id"]}
        documents.append(Document(page_content=content, metadata=metadata))
    
    return documents

def load_llm(model_name='other'):
    if model_name == 'openai':
        if not os.getenv("OPENAI_API_KEY"):
            print("ERROR: Please set your OPENAI_API_KEY as an environment variable.")
            exit(1)

        print("Initializing ML Debugging Assistant...")
    
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # gpt-4o-mini is fast and cheap
    
    else:
        
        # model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        model_id = "distilgpt2"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)

        pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
        )

        llm = HuggingFacePipeline(pipeline=pipe)

    return llm
