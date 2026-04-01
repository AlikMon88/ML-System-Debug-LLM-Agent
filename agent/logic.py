from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def route_query(query: str, llm) -> str:
    ## Categorizes the user's query
    
    system_prompt = """You are an intelligent routing agent for an ML system.
    Categorize the user's issue into exactly ONE of these categories:
    - [COMPUTE] (e.g., OOM, CUDA errors, hardware)
    - [DATA] (e.g., NaN loss, data drift, distribution shift)
    -[CODE] (e.g., syntax errors, tensor shape mismatches)
    
    Respond with ONLY the category tag (e.g., [DATA])."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    
    ## LCEL Chain: Prompt -> LLM -> String Output
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query}).strip()