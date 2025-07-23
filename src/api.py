from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import setup_rag_pipeline

app = FastAPI()
rag_chain = setup_rag_pipeline()

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    result = rag_chain({"question": query.question})
    return {"answer": result["answer"]}