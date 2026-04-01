import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document

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